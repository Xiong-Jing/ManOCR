from typing import Dict

import torch
import torch.nn as nn
from torchvision import models

from manchu_ocr.models.registry import BACKBONES


@BACKBONES.register("MobileNetV3Backbone")
class MobileNetV3Backbone(nn.Module):
    """MobileNetV3 feature extractor returning c2-c5 style feature maps."""

    def __init__(self, arch: str = "large", pretrained: bool = False):
        super().__init__()

        if arch == "large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_large(weights=weights)
            split_indices = [3, 6, 12, len(net.features)]
            self.out_channels = [24, 40, 112, 960]
        elif arch == "small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
            split_indices = [2, 4, 9, len(net.features)]
            self.out_channels = [16, 24, 48, 576]
        else:
            raise ValueError(f"Unsupported MobileNetV3 arch: {arch}")

        features = list(net.features.children())
        self.stage1 = nn.Sequential(*features[: split_indices[0]])
        self.stage2 = nn.Sequential(*features[split_indices[0] : split_indices[1]])
        self.stage3 = nn.Sequential(*features[split_indices[1] : split_indices[2]])
        self.stage4 = nn.Sequential(*features[split_indices[2] : split_indices[3]])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
