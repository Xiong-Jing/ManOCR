from typing import Literal

import torch
import torch.nn as nn

from manchu_ocr.models.recognition.modules.learnable_sobel import LearnableSobel


class DiacriticAwareBranch(nn.Module):
    """
    Fine-grained edge branch for Manchu recognition.

    It extracts local high-frequency details such as dots, circles,
    and small stroke differences.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 192,
        sequence_axis: Literal["height", "width"] = "height",
        dropout: float = 0.1,
        sobel_trainable: bool = True,
        sequence_downsample: int = 4,
        normalize_edges: bool = False,
        edge_scale: float = 1.0,
    ):
        super().__init__()

        if sequence_axis not in {"height", "width"}:
            raise ValueError(f"Unsupported sequence_axis: {sequence_axis}")

        self.sequence_axis = sequence_axis
        self.normalize_edges = normalize_edges
        self.edge_scale = float(edge_scale)
        self.sobel = LearnableSobel(
            in_channels=in_channels,
            trainable=sobel_trainable,
        )

        c1 = embed_dim // 4
        c2 = embed_dim // 2
        c3 = embed_dim

        if sequence_axis == "height":
            if sequence_downsample <= 2:
                # Match the project SVTR patch_size=[2, 16]: preserve more
                # vertical detail while reducing width aggressively.
                strides = [(2, 2), (1, 2), (1, 2), (1, 2)]
            else:
                strides = [(2, 2), (2, 2), (1, 2), (1, 2)]
        else:
            if sequence_downsample <= 2:
                strides = [(2, 2), (2, 1), (2, 1), (2, 1)]
            else:
                strides = [(2, 2), (2, 2), (2, 1), (2, 1)]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, c1, kernel_size=3, stride=strides[0], padding=1, bias=False),
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

            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            detail sequence: [B, T, C]
        """
        edge = self.sobel(x)

        if self.normalize_edges:
            # Sobel magnitudes on normalized images can be much larger than the
            # image tensor itself. Keep the detail branch informative without
            # letting early noisy edge activations dominate the fusion module.
            edge = torch.log1p(edge)
            mean = edge.mean(dim=(-2, -1), keepdim=True)
            std = edge.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            edge = (edge - mean) / std

        edge = edge * self.edge_scale
        x = torch.cat([x, edge], dim=1)

        feat = self.encoder(x)

        if self.sequence_axis == "height":
            seq = feat.mean(dim=3)
        else:
            seq = feat.mean(dim=2)

        seq = seq.transpose(1, 2).contiguous()
        return seq
