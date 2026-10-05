from __future__ import annotations

from typing import Dict, List, Sequence


def restore_polygon(points: Sequence[Sequence[float]], meta: Dict) -> List[List[float]]:
    """Map polygon points from resized/padded image coordinates back to original image."""
    scale_x = float(meta.get("scale_x", 1.0))
    scale_y = float(meta.get("scale_y", 1.0))
    pad_x = float(meta.get("pad_x", 0.0))
    pad_y = float(meta.get("pad_y", 0.0))
    orig_w = float(meta.get("orig_width", meta.get("target_width", 0)))
    orig_h = float(meta.get("orig_height", meta.get("target_height", 0)))

    restored = []
    for x, y in points:
        rx = (float(x) - pad_x) / max(scale_x, 1e-12)
        ry = (float(y) - pad_y) / max(scale_y, 1e-12)
        if orig_w > 0:
            rx = min(max(rx, 0.0), orig_w - 1.0)
        if orig_h > 0:
            ry = min(max(ry, 0.0), orig_h - 1.0)
        restored.append([rx, ry])

    return restored


def restore_polygons(polygons: Sequence[Sequence[Sequence[float]]], meta: Dict) -> List[List[List[float]]]:
    return [restore_polygon(poly, meta) for poly in polygons]
