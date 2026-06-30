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
        as_prob_map: optional auxiliary AS map [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int = 256,
        k: int = 50,
        as_auxiliary_head: bool = False,
    ):
        super().__init__()

        self.k = k
        self.as_auxiliary_head = bool(as_auxiliary_head)
        mid_channels = in_channels // 4

        self.binarize = self._make_map_head(in_channels, mid_channels)
        self.thresh = self._make_map_head(in_channels, mid_channels)
        self.as_binarize = (
            self._make_map_head(in_channels, mid_channels)
            if self.as_auxiliary_head
            else None
        )

    @staticmethod
    def _make_map_head(in_channels: int, mid_channels: int) -> nn.Sequential:
        return nn.Sequential(
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

        outputs = {
            "prob_map": prob_map,
            "thresh_map": thresh_map,
            "binary_map": binary_map,
        }

        if self.as_binarize is not None:
            outputs["as_prob_map"] = self.as_binarize(x)

        return outputs
