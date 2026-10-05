from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from manchu_ocr.models.detection.backbones.resnet import ResNetBackbone
from manchu_ocr.models.detection.heads.db_head import DBHead
from manchu_ocr.models.registry import DETECTORS


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
    ):
        super().__init__()

        if padding is None:
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


class HierarchicalMaskFusion(nn.Module):
    """
    Hi-SAM-style hierarchical text segmentation neck.

    The real Hi-SAM produces masks at pixel/word/line/paragraph levels. This
    project-native baseline keeps that core idea while emitting a DB-compatible
    stride-4 feature map for the shared training and evaluation protocol.
    """

    def __init__(
        self,
        in_channels: list[int],
        inner_channels: int = 160,
        hierarchy_channels: int = 96,
        out_channels: int = 256,
        gate_strength: float = 0.35,
    ):
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(f"in_channels should contain 4 values, got {in_channels}")

        self.gate_strength = float(gate_strength)

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, inner_channels, kernel_size=1, bias=False) for ch in in_channels]
        )

        self.top_down_smooth = nn.ModuleList(
            [ConvBNReLU(inner_channels, inner_channels) for _ in in_channels]
        )

        self.scale_proj = nn.ModuleList(
            [
                ConvBNReLU(inner_channels, hierarchy_channels, kernel_size=3)
                for _ in in_channels
            ]
        )

        self.scale_attention = nn.Sequential(
            ConvBNReLU(hierarchy_channels * 4, hierarchy_channels, kernel_size=3),
            nn.Conv2d(hierarchy_channels, 4, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.word_gate = nn.Sequential(
            ConvBNReLU(hierarchy_channels, hierarchy_channels, kernel_size=3),
            nn.Conv2d(hierarchy_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.line_gate = nn.Sequential(
            ConvBNReLU(hierarchy_channels, hierarchy_channels, kernel_size=5),
            nn.Conv2d(hierarchy_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.paragraph_gate = nn.Sequential(
            ConvBNReLU(hierarchy_channels, hierarchy_channels, kernel_size=7),
            nn.Conv2d(hierarchy_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            ConvBNReLU(hierarchy_channels * 2, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        feats = [features["c2"], features["c3"], features["c4"], features["c5"]]
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]

        p5 = laterals[3]
        p4 = laterals[2] + F.interpolate(p5, size=laterals[2].shape[-2:], mode="nearest")
        p3 = laterals[1] + F.interpolate(p4, size=laterals[1].shape[-2:], mode="nearest")
        p2 = laterals[0] + F.interpolate(p3, size=laterals[0].shape[-2:], mode="nearest")

        pyramids = [
            smooth(feat)
            for smooth, feat in zip(self.top_down_smooth, [p2, p3, p4, p5])
        ]

        target_size = pyramids[0].shape[-2:]
        projected = [
            proj(F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False))
            for proj, feat in zip(self.scale_proj, pyramids)
        ]

        concat = torch.cat(projected, dim=1)
        weights = self.scale_attention(concat)

        fused = 0.0
        for idx, feat in enumerate(projected):
            fused = fused + feat * weights[:, idx : idx + 1]

        word = self.word_gate(fused)
        line = self.line_gate(fused)
        paragraph = self.paragraph_gate(fused)

        hierarchy_gate = (word + line + paragraph) / 3.0
        refined = fused * (1.0 + self.gate_strength * (hierarchy_gate - 0.5))

        return self.out(torch.cat([fused, refined], dim=1))


@DETECTORS.register("HiSAMDetector")
class HiSAMDetector(nn.Module):
    """
    Project-native Hi-SAM comparison detector.

    This is not the external official Hi-SAM repository. It is a trainable
    internal baseline that approximates Hi-SAM's hierarchical mask fusion under
    the project's DB-compatible supervision.
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
            raise ValueError(f"HiSAMDetector expects ResNetBackbone, got {backbone_name}")

        self.backbone = ResNetBackbone(**backbone)

        neck = neck.copy()
        neck.setdefault("in_channels", self.backbone.out_channels)
        self.neck = HierarchicalMaskFusion(**neck)
        self.head = DBHead(**head)

    def forward(self, images: torch.Tensor) -> dict:
        features = self.backbone(images)
        fused = self.neck(features)
        return self.head(fused)
