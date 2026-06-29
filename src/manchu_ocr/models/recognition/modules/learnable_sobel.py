import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSobel(nn.Module):
    """
    Learnable Sobel edge extractor.

    It is initialized with Sobel kernels, but the kernels are trainable.
    This module highlights small dots, circles, and stroke boundaries.
    """

    def __init__(self, in_channels: int = 3, trainable: bool = True):
        super().__init__()

        self.in_channels = in_channels

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        )

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]]
        )

        weight = torch.zeros(2 * in_channels, 1, 3, 3)

        for c in range(in_channels):
            weight[2 * c, 0] = sobel_x
            weight[2 * c + 1, 0] = sobel_y

        self.weight = nn.Parameter(weight, requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            edge map: [B, C, H, W]
        """
        edge = F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=1,
            padding=1,
            groups=self.in_channels,
        )

        b, _, h, w = edge.shape
        edge = edge.view(b, self.in_channels, 2, h, w)

        gx = edge[:, :, 0]
        gy = edge[:, :, 1]

        edge_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        return edge_mag
