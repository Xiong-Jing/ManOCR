import torch
import torch.nn as nn
import torch.nn.functional as F

from manchu_ocr.models.detection.heads.db_head import DBHead
from manchu_ocr.models.registry import DETECTORS


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CRAFTEncoder(nn.Module):
    """
    Lightweight VGG-style encoder used by the CRAFT-style baseline.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, c1),
            ConvBNReLU(c1, c1),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(c1, c2),
            ConvBNReLU(c2, c2),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(c2, c3),
            ConvBNReLU(c3, c3),
        )
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(c3, c4),
            ConvBNReLU(c4, c4),
        )
        self.stage5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(c4, c4),
            ConvBNReLU(c4, c4),
        )

        self.out_channels = [c1, c2, c3, c4, c4]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c1, c2, c3, c4, c5]


class CRAFTDecoder(nn.Module):
    """
    CRAFT-style U-Net decoder with skip connections.

    The decoder returns a stride-4 feature map so the shared DBHead can upsample
    it back to image resolution.
    """

    def __init__(
        self,
        encoder_channels: list[int],
        inner_channels: int = 128,
        out_channels: int = 256,
    ):
        super().__init__()

        c1, c2, c3, c4, c5 = encoder_channels

        self.reduce5 = ConvBNReLU(c5, inner_channels, kernel_size=1)
        self.up4 = ConvBNReLU(inner_channels + c4, inner_channels)
        self.up3 = ConvBNReLU(inner_channels + c3, inner_channels)

        self.out = nn.Sequential(
            ConvBNReLU(inner_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        _, _, c3, c4, c5 = features

        x = self.reduce5(c5)

        x = F.interpolate(x, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, c4], dim=1))

        x = F.interpolate(x, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, c3], dim=1))

        return self.out(x)


@DETECTORS.register("CRAFTDetector")
class CRAFTDetector(nn.Module):
    """
    Project-native CRAFT-style detection baseline.

    It uses a VGG/U-Net style encoder-decoder with skip connections and a
    shared DB-compatible output head for the common evaluation protocol.
    """

    def __init__(
        self,
        backbone: dict,
        neck: dict,
        head: dict,
    ):
        super().__init__()

        backbone = backbone.copy()
        backbone_name = backbone.pop("name", "CRAFTEncoder")

        if backbone_name != "CRAFTEncoder":
            raise ValueError(f"CRAFTDetector expects CRAFTEncoder, got {backbone_name}")

        self.backbone = CRAFTEncoder(**backbone)

        neck = neck.copy()
        neck.setdefault("encoder_channels", self.backbone.out_channels)
        self.neck = CRAFTDecoder(**neck)

        self.head = DBHead(**head)

    def forward(self, images: torch.Tensor) -> dict:
        features = self.backbone(images)
        fused = self.neck(features)
        return self.head(fused)
