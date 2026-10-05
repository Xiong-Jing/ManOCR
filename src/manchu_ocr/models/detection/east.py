import torch
import torch.nn as nn
import torch.nn.functional as F

from manchu_ocr.models.detection.backbones.resnet import ResNetBackbone
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


class EASTFusion(nn.Module):
    """
    EAST-style top-down feature fusion.

    The original EAST detector fuses deep semantic features back to a
    high-resolution map. This implementation keeps the same high-level idea
    while returning a stride-4 feature map compatible with the shared detection
    training and evaluation pipeline.
    """

    def __init__(
        self,
        in_channels: list[int],
        inner_channels: int = 128,
        out_channels: int = 256,
    ):
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(f"in_channels should contain 4 values, got {in_channels}")

        self.proj2 = nn.Conv2d(in_channels[0], inner_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(in_channels[1], inner_channels, kernel_size=1)
        self.proj4 = nn.Conv2d(in_channels[2], inner_channels, kernel_size=1)
        self.proj5 = nn.Conv2d(in_channels[3], inner_channels, kernel_size=1)

        self.fuse4 = ConvBNReLU(inner_channels * 2, inner_channels)
        self.fuse3 = ConvBNReLU(inner_channels * 2, inner_channels)
        self.fuse2 = ConvBNReLU(inner_channels * 2, inner_channels)

        self.out = nn.Sequential(
            ConvBNReLU(inner_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        p2 = self.proj2(features["c2"])
        p3 = self.proj3(features["c3"])
        p4 = self.proj4(features["c4"])
        p5 = self.proj5(features["c5"])

        x = F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse4(torch.cat([x, p4], dim=1))

        x = F.interpolate(x, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse3(torch.cat([x, p3], dim=1))

        x = F.interpolate(x, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, p2], dim=1))

        return self.out(x)


@DETECTORS.register("EASTDetector")
class EASTDetector(nn.Module):
    """
    Project-native EAST-style detection baseline.

    It uses EAST-style feature fusion and the shared DB-compatible output head
    so it can be trained and evaluated with the same scripts as DBNet/DBNet++.
    """

    def __init__(
        self,
        backbone: dict,
        neck: dict,
        head: dict,
    ):
        super().__init__()

        backbone = backbone.copy()
        backbone_name = backbone.pop("name", "ResNetBackbone")

        if backbone_name != "ResNetBackbone":
            raise ValueError(f"EASTDetector currently supports ResNetBackbone, got {backbone_name}")

        self.backbone = ResNetBackbone(**backbone)

        neck = neck.copy()
        if "in_channels" not in neck:
            neck["in_channels"] = self.backbone.out_channels
        self.neck = EASTFusion(**neck)

        self.head = DBHead(**head)

    def forward(self, images: torch.Tensor) -> dict:
        features = self.backbone(images)
        fused = self.neck(features)
        return self.head(fused)
