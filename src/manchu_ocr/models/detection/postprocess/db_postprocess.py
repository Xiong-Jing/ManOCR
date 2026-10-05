from __future__ import annotations

from typing import Dict, List, Tuple

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
    return float(np.sqrt((diff * diff).sum(axis=1)).sum())


def unclip_polygon(points: np.ndarray, unclip_ratio: float = 1.5) -> np.ndarray | None:
    area = polygon_area(points)
    perimeter = polygon_perimeter(points)

    if area <= 1.0 or perimeter <= 1.0:
        return None

    distance = area * unclip_ratio / perimeter

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(
        points.astype(np.int64).tolist(),
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON,
    )

    expanded = pco.Execute(distance)

    if len(expanded) == 0:
        return None

    expanded = sorted(
        expanded,
        key=lambda x: polygon_area(np.asarray(x, dtype=np.float32)),
        reverse=True,
    )

    expanded = np.asarray(expanded[0], dtype=np.float32)

    if expanded.shape[0] < 3:
        return None

    return expanded


def get_mini_boxes(contour: np.ndarray) -> Tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    box = sorted(box, key=lambda x: x[0])

    left = box[:2]
    right = box[2:]

    left = sorted(left, key=lambda x: x[1])
    right = sorted(right, key=lambda x: x[1])

    ordered = np.array(
        [
            left[0],
            right[0],
            right[1],
            left[1],
        ],
        dtype=np.float32,
    )

    short_side = min(rect[1])

    return ordered, float(short_side)


def box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    h, w = bitmap.shape[:2]

    box = box.copy()

    x_min = max(int(np.floor(box[:, 0].min())), 0)
    x_max = min(int(np.ceil(box[:, 0].max())), w - 1)
    y_min = max(int(np.floor(box[:, 1].min())), 0)
    y_max = min(int(np.ceil(box[:, 1].max())), h - 1)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)

    box[:, 0] -= x_min
    box[:, 1] -= y_min

    cv2.fillPoly(mask, [box.astype(np.int32)], 1)

    region = bitmap[y_min:y_max + 1, x_min:x_max + 1]

    if mask.sum() <= 0:
        return 0.0

    return float(cv2.mean(region, mask)[0])


class DBPostProcessor:
    """
    Convert DBNet probability map to detection boxes.
    """

    def __init__(
        self,
        binary_thresh: float = 0.3,
        box_thresh: float = 0.5,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        min_size: int = 3,
    ):
        self.binary_thresh = binary_thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    def __call__(
        self,
        preds: Dict,
    ) -> List[Dict]:
        """
        Args:
            preds:
                model outputs. Uses preds["prob_map"], shape [B, 1, H, W].

        Returns:
            List of per-image dict:
            [
                {
                    "boxes": List[[[x,y], ...]],
                    "scores": List[float]
                }
            ]
        """
        prob_maps = preds["prob_map"].detach().cpu().numpy()

        results = []

        for prob in prob_maps:
            prob = prob[0]
            boxes, scores = self._process_single(prob)

            results.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                }
            )

        return results

    def _process_single(self, prob_map: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        height, width = prob_map.shape

        bitmap = (prob_map > self.binary_thresh).astype(np.uint8)

        contours, _ = cv2.findContours(
            bitmap,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = contours[: self.max_candidates]

        boxes = []
        scores = []

        for contour in contours:
            if contour.shape[0] <= 2:
                continue

            box, short_side = get_mini_boxes(contour)

            if short_side < self.min_size:
                continue

            score = box_score_fast(prob_map, box)

            if score < self.box_thresh:
                continue

            expanded = unclip_polygon(box, self.unclip_ratio)

            if expanded is None:
                continue

            expanded_box, expanded_short_side = get_mini_boxes(expanded.astype(np.float32))

            if expanded_short_side < self.min_size:
                continue

            expanded_box[:, 0] = np.clip(expanded_box[:, 0], 0, width - 1)
            expanded_box[:, 1] = np.clip(expanded_box[:, 1], 0, height - 1)

            boxes.append(expanded_box.tolist())
            scores.append(float(score))

        return boxes, scores
