from typing import Dict

import torch
import torch.nn as nn
from torchvision import models
from manchu_ocr.models.registry import BACKBONES


@BACKBONES.register("ResNetBackbone")
class ResNetBackbone(nn.Module):
    """
    ResNet backbone for DBNet-style text detection.

    Outputs:
        c2: stride 4
        c3: stride 8
        c4: stride 16
        c5: stride 32
    """

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = False,
    ):
        super().__init__()

        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            self.out_channels = [64, 128, 256, 512]
        elif arch == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            net = models.resnet34(weights=weights)
            self.out_channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {arch}")

        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
        )

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
        }
