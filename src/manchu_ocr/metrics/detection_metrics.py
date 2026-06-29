from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def polygon_to_bbox(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=np.float32)

    x1 = float(pts[:, 0].min())
    y1 = float(pts[:, 1].min())
    x2 = float(pts[:, 0].max())
    y2 = float(pts[:, 1].max())

    return x1, y1, x2, y2


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


class DetectionMetric:
    """
    Simple detection metric using bbox IoU.

    Suitable for current LabelMe rectangle annotations.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.num_gt = 0
        self.num_pred = 0

    def update(
        self,
        pred_boxes: List[Sequence[Sequence[float]]],
        gt_polygons: List[Dict],
    ) -> Dict[str, float]:
        gt_boxes = [
            polygon_to_bbox(poly["points"])
            for poly in gt_polygons
        ]

        pred_bboxes = [
            polygon_to_bbox(box)
            for box in pred_boxes
        ]

        self.num_gt += len(gt_boxes)
        self.num_pred += len(pred_bboxes)

        matched_gt = set()

        image_tp = 0
        image_fp = 0

        for pred_box in pred_bboxes:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = bbox_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                image_tp += 1
            else:
                image_fp += 1

        image_fn = len(gt_boxes) - len(matched_gt)

        self.tp += image_tp
        self.fp += image_fp
        self.fn += image_fn

        return {
            "tp": image_tp,
            "fp": image_fp,
            "fn": image_fn,
            "num_gt": len(gt_boxes),
            "num_pred": len(pred_bboxes),
        }

    def compute(self) -> Dict[str, float]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)

        if precision + recall <= 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "fmeasure": float(fmeasure),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "num_gt": int(self.num_gt),
            "num_pred": int(self.num_pred),
        }
