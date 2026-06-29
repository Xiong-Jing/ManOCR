import torch
import torch.nn as nn

from manchu_ocr.models.registry import HEADS


@HEADS.register("DBHead")
class DBHead(nn.Module):
    """
    Differentiable Binarization head.

    Input:
        FPN feature map, stride 4

    Output:
        prob_map:   [B, 1, H, W]
        thresh_map: [B, 1, H, W]
        binary_map: [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int = 256,
        k: int = 50,
    ):
        super().__init__()

        self.k = k
        mid_channels = in_channels // 4

        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(mid_channels, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(mid_channels, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        prob_map = self.binarize(x)
        thresh_map = self.thresh(x)

        binary_map = torch.reciprocal(
            1.0 + torch.exp(-self.k * (prob_map - thresh_map))
        )

        return {
            "prob_map": prob_map,
            "thresh_map": thresh_map,
            "binary_map": binary_map,
        }
