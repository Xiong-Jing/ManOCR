from typing import Literal

import torch
import torch.nn as nn


class SequencePatchEmbed(nn.Module):
    """
    Convert image feature map into a 1D sequence for CTC recognition.

    For vertical Manchu text:
        sequence_axis = "height"
        input [B, 3, 256, 64] -> sequence length about 64

    For horizontal text:
        sequence_axis = "width"
        input [B, 3, 64, 256] -> sequence length about 64
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 192,
        sequence_axis: Literal["height", "width"] = "height",
    ):
        super().__init__()

        if sequence_axis not in {"height", "width"}:
            raise ValueError(f"Unsupported sequence_axis: {sequence_axis}")

        self.sequence_axis = sequence_axis

        c1 = embed_dim // 4
        c2 = embed_dim // 2
        c3 = embed_dim

        if sequence_axis == "height":
            # Preserve vertical resolution, aggressively reduce width.
            strides = [(2, 2), (2, 2), (1, 2), (1, 2)]
        else:
            # Preserve horizontal resolution, aggressively reduce height.
            strides = [(2, 2), (2, 2), (2, 1), (2, 1)]

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=strides[0], padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),

            nn.Conv2d(c1, c2, kernel_size=3, stride=strides[1], padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),

            nn.Conv2d(c2, c3, kernel_size=3, stride=strides[2], padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),

            nn.Conv2d(c3, embed_dim, kernel_size=3, stride=strides[3], padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            sequence: [B, T, C]
        """
        feat = self.proj(x)  # [B, C, H', W']

        if self.sequence_axis == "height":
            # Average over width, keep height as sequence.
            seq = feat.mean(dim=3)      # [B, C, H']
        else:
            # Average over height, keep width as sequence.
            seq = feat.mean(dim=2)      # [B, C, W']

        seq = seq.transpose(1, 2).contiguous()  # [B, T, C]
        return seq
0