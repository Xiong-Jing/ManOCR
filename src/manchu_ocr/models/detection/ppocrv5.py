from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from manchu_ocr.models.detection.backbones.mobilenetv3 import MobileNetV3Backbone
from manchu_ocr.models.detection.heads.db_head import DBHead
from manchu_ocr.models.registry import DETECTORS


class ConvBNHSwish(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        self.block = nn.Sequential(
            ConvBNHSwish(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels),
            ConvBNHSwish(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LiteCoordinateAttention(nn.Module):
    """
    Lightweight coordinate attention used to mimic PP-OCR style mobile heads.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        hidden = max(channels // reduction, 32)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
        )
        self.attn_h = nn.Conv2d(hidden, channels, kernel_size=1)
        self.attn_w = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        context_h = x.mean(dim=3, keepdim=True)
        context_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        context = torch.cat([context_h, context_w], dim=2)
        context = self.shared(context)

        attn_h, attn_w = torch.split(context, [h, w], dim=2)
        attn_w = attn_w.transpose(2, 3)

        weight = torch.sigmoid(self.attn_h(attn_h)) * torch.sigmoid(self.attn_w(attn_w))
        return x * weight


class PPOCRv5PAN(nn.Module):
    """
    PP-OCRv5-style lightweight PAN neck.

    It follows the industrial OCR design preference for MobileNet features,
    depthwise separable convolutions, coordinate attention, and a DB-compatible
    output feature map.
    """

    def __init__(
        self,
        in_channels: list[int],
        inner_channels: int = 128,
        out_channels: int = 256,
        attention_reduction: int = 8,
    ):
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(f"in_channels should contain 4 values, got {in_channels}")

        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, inner_channels, kernel_size=1, bias=False) for ch in in_channels]
        )

        self.top_down = nn.ModuleList(
            [DepthwiseSeparableConv(inner_channels, inner_channels) for _ in range(3)]
        )

        self.bottom_up_downsample = nn.ModuleList(
            [
                DepthwiseSeparableConv(inner_channels, inner_channels, kernel_size=3),
                DepthwiseSeparableConv(inner_channels, inner_channels, kernel_size=3),
            ]
        )

        self.attention = LiteCoordinateAttention(
            channels=inner_channels * 4,
            reduction=attention_reduction,
        )
        self.out = nn.Sequential(
            DepthwiseSeparableConv(inner_channels * 4, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        feats = [features["c2"], features["c3"], features["c4"], features["c5"]]
        c2, c3, c4, c5 = [conv(feat) for conv, feat in zip(self.lateral, feats)]

        p5 = c5
        p4 = self.top_down[0](c4 + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.top_down[1](c3 + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        p2 = self.top_down[2](c2 + F.interpolate(p3, size=c2.shape[-2:], mode="nearest"))

        # A lightweight bottom-up path keeps PP-OCR-like PAN behavior without
        # changing the final stride expected by DBHead.
        n3 = p3 + F.interpolate(p2, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        n4 = p4 + F.interpolate(n3, size=p4.shape[-2:], mode="bilinear", align_corners=False)

        target_size = p2.shape[-2:]
        maps = [
            p2,
            F.interpolate(n3, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(n4, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(p5, size=target_size, mode="bilinear", align_corners=False),
        ]

        fused = torch.cat(maps, dim=1)
        fused = self.attention(fused)
        return self.out(fused)


@DETECTORS.register("PPOCRv5Detector")
class PPOCRv5Detector(nn.Module):
    """
    Project-native PP-OCRv5 detection comparison baseline.

    This approximates the PP-OCRv5 detection design using a MobileNetV3
    backbone, lightweight PAN neck, coordinate attention, and DB-style head.
    """

    def __init__(
        self,
        backbone: dict,
        neck: dict,
        head: dict,
    ):
        super().__init__()

        backbone = backbone.copy()
        backbone_name = backbone.pop("name", "MobileNetV3Backbone")

        if backbone_name != "MobileNetV3Backbone":
            raise ValueError(f"PPOCRv5Detector expects MobileNetV3Backbone, got {backbone_name}")

        self.backbone = MobileNetV3Backbone(**backbone)

        neck = neck.copy()
        neck.setdefault("in_channels", self.backbone.out_channels)
        self.neck = PPOCRv5PAN(**neck)
        self.head = DBHead(**head)

    def forward(self, images: torch.Tensor) -> dict:
        features = self.backbone(images)
        fused = self.neck(features)
        return self.head(fused)
