from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from manchu_ocr.models.detection.modules.vsaa import VSAA


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__(
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


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


class StripPoolingAttention(nn.Module):
    """Canonical two-branch strip-pooling counterpart.

    The layout follows the Strip Pooling Module from Hou et al. (CVPR 2020):
    one branch mixes two ordinary adaptive-pooling grids, while the other
    combines horizontal and vertical global strips. The module is inserted at
    exactly the same post-FPN location as VSAA and preserves the input shape.
    """

    def __init__(
        self,
        channels: int,
        pool_sizes: Sequence[Sequence[int]] = ((20, 12), (12, 20)),
    ) -> None:
        super().__init__()

        if len(pool_sizes) != 2 or any(len(size) != 2 for size in pool_sizes):
            raise ValueError(
                "pool_sizes must contain two (height, width) pairs, "
                f"got {pool_sizes!r}"
            )

        normalized_pool_sizes = tuple(
            (int(size[0]), int(size[1])) for size in pool_sizes
        )
        if any(value <= 0 for size in normalized_pool_sizes for value in size):
            raise ValueError(f"pool_sizes must be positive, got {pool_sizes!r}")

        hidden_channels = max(int(channels) // 4, 1)

        self.pool1 = nn.AdaptiveAvgPool2d(normalized_pool_sizes[0])
        self.pool2 = nn.AdaptiveAvgPool2d(normalized_pool_sizes[1])

        self.branch1_reduce = ConvBNReLU(channels, hidden_channels, kernel_size=1)
        self.branch2_reduce = ConvBNReLU(channels, hidden_channels, kernel_size=1)

        self.branch1_local = ConvBN(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.branch1_pool1 = ConvBN(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.branch1_pool2 = ConvBN(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

        self.horizontal_strip = ConvBN(
            hidden_channels,
            hidden_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
        )
        self.vertical_strip = ConvBN(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
        )

        self.branch1_out = ConvBNReLU(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.branch2_out = ConvBNReLU(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.out_proj = ConvBN(hidden_channels * 2, channels, kernel_size=1)

    @staticmethod
    def _resize(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            feature,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        output_size = (height, width)

        branch1 = self.branch1_reduce(x)
        branch2 = self.branch2_reduce(x)

        branch1_context = self.branch1_local(branch1)
        branch1_context = branch1_context + self._resize(
            self.branch1_pool1(self.pool1(branch1)),
            output_size,
        )
        branch1_context = branch1_context + self._resize(
            self.branch1_pool2(self.pool2(branch1)),
            output_size,
        )
        branch1_context = self.branch1_out(F.relu(branch1_context, inplace=False))

        # A horizontal 1xW strip aggregates width and retains row position.
        horizontal_context = branch2.mean(dim=3, keepdim=True)
        horizontal_context = self.horizontal_strip(horizontal_context)
        horizontal_context = self._resize(horizontal_context, output_size)

        # A vertical Hx1 strip aggregates height and retains column position.
        vertical_context = branch2.mean(dim=2, keepdim=True)
        vertical_context = self.vertical_strip(vertical_context)
        vertical_context = self._resize(vertical_context, output_size)

        branch2_context = self.branch2_out(
            F.relu(horizontal_context + vertical_context, inplace=False)
        )
        residual = self.out_proj(
            torch.cat([branch1_context, branch2_context], dim=1)
        )
        return F.relu(x + residual, inplace=False)


class SingleDirectionStripAttention(nn.Module):
    """One-direction strip counterpart with the same post-FPN interface.

    ``orientation='horizontal'`` uses a 1xW strip (averages over width), while
    ``orientation='vertical'`` uses an Hx1 strip (averages over height).
    """

    def __init__(
        self,
        channels: int,
        orientation: str,
        reduction: int = 4,
    ) -> None:
        super().__init__()

        orientation = str(orientation).strip().lower()
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError(
                "orientation must be 'horizontal' or 'vertical', "
                f"got {orientation!r}"
            )
        if int(reduction) <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")

        hidden_channels = max(int(channels) // int(reduction), 1)
        self.orientation = orientation
        self.reduce = ConvBNReLU(channels, hidden_channels, kernel_size=1)

        if orientation == "horizontal":
            kernel_size = (3, 1)
            padding = (1, 0)
        else:
            kernel_size = (1, 3)
            padding = (0, 1)

        self.context = ConvBN(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.out_proj = ConvBN(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)

        if self.orientation == "horizontal":
            context = reduced.mean(dim=3, keepdim=True)
        else:
            context = reduced.mean(dim=2, keepdim=True)

        context = self.context(context)
        context = F.interpolate(
            context,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual = self.out_proj(F.relu(context, inplace=False))
        return F.relu(x + residual, inplace=False)


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3.0) / 6.0


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.gate = HardSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class CoordinateAttention(nn.Module):
    """Coordinate Attention counterpart following the authors' reference code."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()

        if int(reduction) <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")

        hidden_channels = max(8, int(channels) // int(reduction))
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            HardSwish(),
        )
        self.height_gate = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.width_gate = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        height_context = x.mean(dim=3, keepdim=True)
        width_context = x.mean(dim=2, keepdim=True).transpose(2, 3)

        context = self.shared(torch.cat([height_context, width_context], dim=2))
        height_context, width_context = torch.split(
            context,
            [height, width],
            dim=2,
        )
        width_context = width_context.transpose(2, 3)

        height_weight = torch.sigmoid(self.height_gate(height_context))
        width_weight = torch.sigmoid(self.width_gate(width_context))
        return x * height_weight * width_weight


_DIRECTION_MODULE_ALIASES = {
    "identity": "none",
    "no_attention": "none",
    "no-attention": "none",
    "strip": "strip_pooling",
    "strip-pooling": "strip_pooling",
    "coordinate": "coordinate_attention",
    "coordatt": "coordinate_attention",
    "coordinate-attention": "coordinate_attention",
    "horizontal-only": "horizontal_strip",
    "horizontal_only": "horizontal_strip",
    "vertical-only": "vertical_strip",
    "vertical_only": "vertical_strip",
}


def normalize_direction_module_name(name: str) -> str:
    normalized = str(name).strip().lower()
    normalized = _DIRECTION_MODULE_ALIASES.get(normalized, normalized)
    supported = {
        "none",
        "strip_pooling",
        "coordinate_attention",
        "horizontal_strip",
        "vertical_strip",
        "vsaa",
    }
    if normalized not in supported:
        raise ValueError(
            f"Unsupported direction_module={name!r}. "
            f"Supported values: {sorted(supported)}"
        )
    return normalized


def build_direction_module(
    name: str,
    *,
    channels: int,
    strip_pool_sizes: Sequence[Sequence[int]] = ((20, 12), (12, 20)),
    strip_reduction: int = 4,
    coordinate_reduction: int = 32,
    vsaa_reduction: int = 4,
    vsaa_vertical_kernel: int = 15,
    vsaa_horizontal_kernel: int = 5,
    vsaa_dropout: float = 0.0,
    vsaa_residual_scale: float = 0.1,
) -> nn.Module:
    normalized = normalize_direction_module_name(name)

    if normalized == "none":
        return nn.Identity()
    if normalized == "strip_pooling":
        return StripPoolingAttention(
            channels=channels,
            pool_sizes=strip_pool_sizes,
        )
    if normalized == "coordinate_attention":
        return CoordinateAttention(
            channels=channels,
            reduction=coordinate_reduction,
        )
    if normalized == "horizontal_strip":
        return SingleDirectionStripAttention(
            channels=channels,
            orientation="horizontal",
            reduction=strip_reduction,
        )
    if normalized == "vertical_strip":
        return SingleDirectionStripAttention(
            channels=channels,
            orientation="vertical",
            reduction=strip_reduction,
        )

    return VSAA(
        channels=channels,
        reduction=vsaa_reduction,
        vertical_kernel=vsaa_vertical_kernel,
        horizontal_kernel=vsaa_horizontal_kernel,
        dropout=vsaa_dropout,
        residual_scale=vsaa_residual_scale,
    )
