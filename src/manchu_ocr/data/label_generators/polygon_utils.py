from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]
Polygon = List[Point]


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """
    Shoelace formula.
    """
    if len(points) < 3:
        return 0.0

    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def polygon_bbox(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=np.float64)
    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    return x_min, y_min, x_max, y_max


def bbox_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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

    inter = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(box_a) + bbox_area(box_b) - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def clip_polygon_to_image(
    points: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
) -> Polygon:
    clipped = []

    for x, y in points:
        x = min(max(float(x), 0.0), float(image_width - 1))
        y = min(max(float(y), 0.0), float(image_height - 1))
        clipped.append((x, y))

    return clipped


def is_valid_polygon(points: Sequence[Sequence[float]], min_area: float = 4.0) -> bool:
    if len(points) < 3:
        return False

    area = polygon_area(points)
    return area >= min_area


def rectangle_to_polygon(points: Sequence[Sequence[float]]) -> Polygon:
    """
    Convert LabelMe rectangle two points to four-point polygon.
    """
    if len(points) != 2:
        raise ValueError(f"Rectangle should contain 2 points, got {len(points)}")

    x1, y1 = points[0]
    x2, y2 = points[1]

    left = min(float(x1), float(x2))
    right = max(float(x1), float(x2))
    top = min(float(y1), float(y2))
    bottom = max(float(y1), float(y2))

    return [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]


def normalize_points(points: Iterable[Iterable[float]]) -> Polygon:
    return [(float(x), float(y)) for x, y in points]
