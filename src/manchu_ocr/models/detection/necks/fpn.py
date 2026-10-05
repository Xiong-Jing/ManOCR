from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from manchu_ocr.models.registry import NECKS


@NECKS.register("FPN")
class FPN(nn.Module):
    """
    Feature Pyramid Network for DBNet baseline.
    """

    def __init__(
        self,
        in_channels: List[int],
        inner_channels: int = 128,
        out_channels: int = 256,
    ):
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(f"in_channels should contain 4 values, got {in_channels}")

        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(c, inner_channels, kernel_size=1)
                for c in in_channels
            ]
        )

        self.smooth_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(inner_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(inner_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        c2 = features["c2"]
        c3 = features["c3"]
        c4 = features["c4"]
        c5 = features["c5"]

        feats = [c2, c3, c4, c5]

        laterals = [
            conv(feat)
            for conv, feat in zip(self.lateral_convs, feats)
        ]

        p5 = laterals[3]
        p4 = laterals[2] + F.interpolate(p5, size=laterals[2].shape[-2:], mode="bilinear", align_corners=False)
        p3 = laterals[1] + F.interpolate(p4, size=laterals[1].shape[-2:], mode="bilinear", align_corners=False)
        p2 = laterals[0] + F.interpolate(p3, size=laterals[0].shape[-2:], mode="bilinear", align_corners=False)

        pyramids = [p2, p3, p4, p5]

        pyramids = [
            smooth(p)
            for smooth, p in zip(self.smooth_convs, pyramids)
        ]

        target_size = pyramids[0].shape[-2:]

        pyramids = [
            F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            for p in pyramids
        ]

        fused = torch.cat(pyramids, dim=1)
        fused = self.out_conv(fused)

        return fused
