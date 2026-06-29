from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def polygon_to_bbox(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=np.float32)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def bbox_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def polygon_iou(poly_a: Sequence[Sequence[float]], poly_b: Sequence[Sequence[float]]) -> float:
    """BBox IoU fallback for polygon annotations used by current experiments."""
    return bbox_iou(polygon_to_bbox(poly_a), polygon_to_bbox(poly_b))
