import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        padding,
        groups: int = 1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,    
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VSAA(nn.Module):
    """
    Vertical Spatial Attention Aggregation module.

    Designed for vertical Manchu document detection.
    
    It enhances:
        - vertical stroke continuity
        - column-wise text structure
        - fine spatial response around narrow word regions
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        vertical_kernel: int = 15,
        horizontal_kernel: int = 5,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
    ):
        super().__init__()

        if vertical_kernel % 2 == 0:
            raise ValueError("vertical_kernel should be odd.")

        if horizontal_kernel % 2 == 0:
            raise ValueError("horizontal_kernel should be odd.")

        hidden_channels = max(channels // reduction, 32)

        self.reduce = ConvBNAct(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding=0,
        )

        self.vertical_context = nn.Sequential(
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=(vertical_kernel, 1),
                padding=(vertical_kernel // 2, 0),
                groups=hidden_channels,
            ),
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
            ),
        )

        self.horizontal_context = nn.Sequential(
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=(1, horizontal_kernel),
                padding=(0, horizontal_kernel // 2),
                groups=hidden_channels,
            ),
            ConvBNAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
            ),
        )

        self.context_fuse = ConvBNAct(
            in_channels=hidden_channels * 3,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out_proj = ConvBNAct(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        reduced = self.reduce(x)

        vertical_feat = self.vertical_context(reduced)
        horizontal_feat = self.horizontal_context(reduced)

        context = torch.cat(
            [
                reduced,
                vertical_feat,
                horizontal_feat,
            ],
            dim=1,
        )

        context = self.context_fuse(context)

        spatial_weight = self.spatial_gate(context)
        channel_weight = self.channel_gate(identity)

        refined = self.out_proj(context)
        refined = refined * spatial_weight * channel_weight
        refined = self.dropout(refined)

        residual_scale = torch.clamp(self.residual_scale, min=0.0, max=1.0)
        return identity + residual_scale * refined
