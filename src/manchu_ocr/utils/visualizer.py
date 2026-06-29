from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def imread_unicode(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def imwrite_unicode(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def draw_polygons(
    image: np.ndarray,
    polygons: Sequence[Sequence[Sequence[float]]],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    canvas = image.copy()
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
    return canvas
