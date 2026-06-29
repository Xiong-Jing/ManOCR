from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from manchu_ocr.models.detection.modules.vsaa import VSAA
from manchu_ocr.models.registry import NECKS


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()

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


class ASFBlock(nn.Module):
    """
    Adaptive Scale Fusion block.

    This is a lightweight DBNet++-style scale attention module.
    It learns spatially varying weights for multi-scale FPN features.
    """

    def __init__(self, channels: int, num_scales: int = 4):
        super().__init__()

        self.num_scales = num_scales

        self.attention = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_scales, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} feature maps, got {len(features)}")

        concat = torch.cat(features, dim=1)
        weights = self.attention(concat)

        weighted_features = []

        for i, feat in enumerate(features):
            weighted_features.append(feat * weights[:, i:i + 1])

        return torch.cat(weighted_features, dim=1)


@NECKS.register("DBFPN")
class DBFPN(nn.Module):
    """
    DBNet/DBNet++ style FPN.

    Input:
        c2, c3, c4, c5

    Output:
        fused feature map at stride 4.
    """

    def __init__(
        self,
        in_channels: List[int],
        inner_channels: int = 256,
        out_channels: int = 256,
        use_asf: bool = True,
        use_vsaa: bool = False,
        vsaa_reduction: int = 4,
        vsaa_vertical_kernel: int = 15,
        vsaa_horizontal_kernel: int = 5,
        vsaa_dropout: float = 0.0,
        vsaa_residual_scale: float = 0.1,
    ):
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(f"in_channels should contain 4 values, got {in_channels}")

        self.use_asf = use_asf

        self.in5 = nn.Conv2d(in_channels[3], inner_channels, kernel_size=1, bias=False)
        self.in4 = nn.Conv2d(in_channels[2], inner_channels, kernel_size=1, bias=False)
        self.in3 = nn.Conv2d(in_channels[1], inner_channels, kernel_size=1, bias=False)
        self.in2 = nn.Conv2d(in_channels[0], inner_channels, kernel_size=1, bias=False)

        self.out5 = ConvBNReLU(inner_channels, inner_channels // 4, kernel_size=3, padding=1)
        self.out4 = ConvBNReLU(inner_channels, inner_channels // 4, kernel_size=3, padding=1)
        self.out3 = ConvBNReLU(inner_channels, inner_channels // 4, kernel_size=3, padding=1)
        self.out2 = ConvBNReLU(inner_channels, inner_channels // 4, kernel_size=3, padding=1)

        scale_channels = inner_channels // 4

        if self.use_asf:
            self.asf = ASFBlock(
                channels=scale_channels,
                num_scales=4,
            )
            fusion_in_channels = scale_channels * 4
        else:
            self.asf = None
            fusion_in_channels = scale_channels * 4

        self.fusion = ConvBNReLU(
            fusion_in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.use_vsaa = use_vsaa

        if self.use_vsaa:
            self.vsaa = VSAA(
                channels=out_channels,
                reduction=vsaa_reduction,
                vertical_kernel=vsaa_vertical_kernel,
                horizontal_kernel=vsaa_horizontal_kernel,
                dropout=vsaa_dropout,
                residual_scale=vsaa_residual_scale,
            )
        else:
            self.vsaa = nn.Identity()

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        c2 = features["c2"]
        c3 = features["c3"]
        c4 = features["c4"]
        c5 = features["c5"]

        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = in4 + F.interpolate(in5, size=in4.shape[-2:], mode="nearest")
        out3 = in3 + F.interpolate(out4, size=in3.shape[-2:], mode="nearest")
        out2 = in2 + F.interpolate(out3, size=in2.shape[-2:], mode="nearest")

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        target_size = p2.shape[-2:]

        p5 = F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)

        multi_scale_features = [p2, p3, p4, p5]

        if self.use_asf:
            fused = self.asf(multi_scale_features)
        else:
            fused = torch.cat(multi_scale_features, dim=1)

        fused = self.fusion(fused)
        fused = self.vsaa(fused)

        return fused
