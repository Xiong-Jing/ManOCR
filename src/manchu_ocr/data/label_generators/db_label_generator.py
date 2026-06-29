from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pyclipper


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0

    diff = points - np.roll(points, -1, axis=0)
    length = np.sqrt((diff * diff).sum(axis=1)).sum()

    return float(length)


def shrink_polygon(points: np.ndarray, shrink_ratio: float) -> np.ndarray | None:
    """
    Shrink polygon for DB probability map.

    DBNet shrink distance:
        D = A * (1 - r^2) / L
    """
    area = polygon_area(points)
    perimeter = polygon_perimeter(points)

    if area <= 1.0 or perimeter <= 1.0:
        return None

    distance = area * (1.0 - shrink_ratio * shrink_ratio) / perimeter

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(points.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    shrinked = pco.Execute(-distance)

    if len(shrinked) == 0:
        return None

    shrinked = sorted(shrinked, key=lambda x: polygon_area(np.asarray(x, dtype=np.float32)), reverse=True)
    shrinked = np.asarray(shrinked[0], dtype=np.float32)

    if len(shrinked) < 3:
        return None

    return shrinked


def asymmetric_shrink_polygon(
    points: np.ndarray,
    shrink_ratio_x: float = 0.65,
    shrink_ratio_y: float = 0.90,
) -> np.ndarray | None:
    """
    Asymmetric shrink for vertical Manchu word boxes.

    For vertical text:
        - shrink more in x direction
        - shrink less in y direction

    This is especially suitable for LabelMe rectangle annotations.
    """
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return None

    shrink_ratio_x = float(np.clip(shrink_ratio_x, 0.05, 1.0))
    shrink_ratio_y = float(np.clip(shrink_ratio_y, 0.05, 1.0))

    x_min = points[:, 0].min()
    y_min = points[:, 1].min()
    x_max = points[:, 0].max()
    y_max = points[:, 1].max()

    width = x_max - x_min
    height = y_max - y_min

    if width <= 1.0 or height <= 1.0:
        return None

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    shrinked = points.copy().astype(np.float32)
    shrinked[:, 0] = cx + (shrinked[:, 0] - cx) * shrink_ratio_x
    shrinked[:, 1] = cy + (shrinked[:, 1] - cy) * shrink_ratio_y

    if polygon_area(shrinked) <= 1.0:
        return None

    return shrinked


def expand_polygon(points: np.ndarray, distance: float) -> np.ndarray | None:
    if len(points) < 3:
        return None

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(points.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    expanded = pco.Execute(distance)

    if len(expanded) == 0:
        return None

    expanded = sorted(expanded, key=lambda x: polygon_area(np.asarray(x, dtype=np.float32)), reverse=True)
    expanded = np.asarray(expanded[0], dtype=np.float32)

    if len(expanded) < 3:
        return None

    return expanded


def distance_point_to_segment(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """
    Compute distance from many points to one line segment.

    Args:
        points: [N, 2]
        start: [2]
        end: [2]
    """
    segment = end - start
    segment_len_sq = float(np.sum(segment * segment))

    if segment_len_sq < 1e-6:
        diff = points - start
        return np.sqrt(np.sum(diff * diff, axis=1))

    t = np.sum((points - start) * segment, axis=1) / segment_len_sq
    t = np.clip(t, 0.0, 1.0)

    projection = start[None, :] + t[:, None] * segment[None, :]
    diff = points - projection

    return np.sqrt(np.sum(diff * diff, axis=1))


def draw_threshold_map_for_polygon(
    polygon: np.ndarray,
    canvas: np.ndarray,
    mask: np.ndarray,
    shrink_ratio: float,
    thresh_min: float,
    thresh_max: float,
) -> None:
    """
    Generate DBNet-style threshold map for one polygon.
    """
    height, width = canvas.shape

    area = polygon_area(polygon)
    perimeter = polygon_perimeter(polygon)

    if area <= 1.0 or perimeter <= 1.0:
        return

    distance = area * (1.0 - shrink_ratio * shrink_ratio) / perimeter

    if distance <= 1.0:
        distance = 1.0

    expanded = expand_polygon(polygon, distance)

    if expanded is None:
        return

    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)

    expanded_int = expanded.astype(np.int32)

    cv2.fillPoly(mask, [expanded_int], 1.0)

    x_min = max(int(np.floor(expanded[:, 0].min())), 0)
    x_max = min(int(np.ceil(expanded[:, 0].max())), width - 1)
    y_min = max(int(np.floor(expanded[:, 1].min())), 0)
    y_max = min(int(np.ceil(expanded[:, 1].max())), height - 1)

    if x_max <= x_min or y_max <= y_min:
        return

    xs = np.arange(x_min, x_max + 1)
    ys = np.arange(y_min, y_max + 1)

    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)

    min_dist = np.full((grid_points.shape[0],), np.inf, dtype=np.float32)

    num_points = polygon.shape[0]

    for i in range(num_points):
        start = polygon[i]
        end = polygon[(i + 1) % num_points]
        dist = distance_point_to_segment(grid_points, start, end)
        min_dist = np.minimum(min_dist, dist.astype(np.float32))

    dist_map = min_dist.reshape((y_max - y_min + 1, x_max - x_min + 1))
    dist_map = 1.0 - np.clip(dist_map / distance, 0.0, 1.0)

    canvas_region = canvas[y_min:y_max + 1, x_min:x_max + 1]
    canvas[y_min:y_max + 1, x_min:x_max + 1] = np.maximum(canvas_region, dist_map)

    canvas *= 1.0


@dataclass
class DBLabelGenerator:
    """
    Generate DBNet supervision maps.

    Output:
        prob_map:       shrunk text region map
        thresh_map:     adaptive threshold map
        thresh_mask:    valid threshold supervision region
        training_mask:  valid probability supervision region
    """

    shrink_ratio: float = 0.4
    thresh_min: float = 0.3
    thresh_max: float = 0.7
    min_text_size: int = 3
    use_asymmetric_shrink: bool = False
    shrink_ratio_x: float = 0.65
    shrink_ratio_y: float = 0.90
    as_auxiliary: bool = True

    def __call__(
        self,
        polygons: List[Dict],
        image_height: int,
        image_width: int,
    ) -> Dict[str, np.ndarray]:
        prob_map = np.zeros((image_height, image_width), dtype=np.float32)
        thresh_map = np.zeros((image_height, image_width), dtype=np.float32)
        thresh_mask = np.zeros((image_height, image_width), dtype=np.float32)
        training_mask = np.ones((image_height, image_width), dtype=np.float32)
        as_prob_map = np.zeros((image_height, image_width), dtype=np.float32)

        valid_polygons = []

        for poly in polygons:
            points = np.asarray(poly["points"], dtype=np.float32)

            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                continue

            points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)

            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)

            box_w = x_max - x_min
            box_h = y_max - y_min

            if min(box_w, box_h) < self.min_text_size:
                cv2.fillPoly(training_mask, [points.astype(np.int32)], 0)
                continue

            shrinked = shrink_polygon(points, self.shrink_ratio)

            if self.use_asymmetric_shrink:
                as_shrinked = asymmetric_shrink_polygon(
                    points,
                    shrink_ratio_x=self.shrink_ratio_x,
                    shrink_ratio_y=self.shrink_ratio_y,
                )

                if as_shrinked is not None:
                    cv2.fillPoly(as_prob_map, [as_shrinked.astype(np.int32)], 1.0)

                if not self.as_auxiliary:
                    shrinked = as_shrinked

            if shrinked is None:
                cv2.fillPoly(training_mask, [points.astype(np.int32)], 0)
                continue

            cv2.fillPoly(prob_map, [shrinked.astype(np.int32)], 1.0)

            draw_threshold_map_for_polygon(
                polygon=points,
                canvas=thresh_map,
                mask=thresh_mask,
                shrink_ratio=self.shrink_ratio,
                thresh_min=self.thresh_min,
                thresh_max=self.thresh_max,
            )

            valid_polygons.append(poly)

        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min

        if self.use_asymmetric_shrink:
            # AS is an auxiliary continuity prior. Keep it compatible with
            # the standard DB shrink target so the main probability head is
            # never asked to suppress standard positive pixels.
            as_prob_map = np.maximum(as_prob_map, prob_map)

        outputs = {
            "prob_map": prob_map,
            "thresh_map": thresh_map,
            "thresh_mask": thresh_mask,
            "training_mask": training_mask,
            "num_valid_polygons": np.asarray([len(valid_polygons)], dtype=np.int64),
        }

        if self.use_asymmetric_shrink:
            outputs["as_prob_map"] = as_prob_map

        return outputs
