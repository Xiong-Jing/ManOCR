from __future__ import annotations

from typing import Sequence

from PIL import Image


def polygon_to_bbox(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def crop_polygon_bbox(
    image: Image.Image,
    points: Sequence[Sequence[float]],
    padding: int = 4,
    padding_ratio: float = 0.04,
) -> Image.Image:
    """Crop an axis-aligned rectangle around a detection polygon."""
    x1, y1, x2, y2 = polygon_to_bbox(points)

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    dynamic_padding = int(round(max(box_w, box_h) * padding_ratio))
    pad = max(int(padding), dynamic_padding)

    x1 = max(0, int(round(x1)) - pad)
    y1 = max(0, int(round(y1)) - pad)
    x2 = min(image.width, int(round(x2)) + pad)
    y2 = min(image.height, int(round(y2)) + pad)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box from polygon: {points}")
    return image.crop((x1, y1, x2, y2))
