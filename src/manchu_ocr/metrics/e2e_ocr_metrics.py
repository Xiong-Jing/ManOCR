from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from manchu_ocr.metrics.detection_metrics import bbox_iou, polygon_to_bbox
from manchu_ocr.metrics.recognition_metrics import levenshtein_error_counts


GENERIC_LABELS = {"", "text", "word", "manchu", "ignore"}


def _normalize_points(points: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points]


def _rectangle_to_polygon(
    points: Sequence[Sequence[float]],
) -> List[List[float]]:
    if len(points) != 2:
        raise ValueError(f"Rectangle should contain 2 points, got {len(points)}")

    x1, y1 = points[0]
    x2, y2 = points[1]
    left = min(float(x1), float(x2))
    right = max(float(x1), float(x2))
    top = min(float(y1), float(y2))
    bottom = max(float(y1), float(y2))
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def extract_transcription(item: Mapping[str, Any]) -> str:
    """Extract a real word transcription, excluding detection-only labels."""
    for key in ("text", "transcription", "label", "value"):
        value = str(item.get(key, "")).strip()
        if value and value.lower() not in GENERIC_LABELS:
            return value
    return ""


def parse_page_annotation(path: str | Path) -> List[Dict[str, Any]]:
    """Load word polygons and transcriptions from a page annotation JSON."""
    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "items" in data:
        source_items = data["items"]
    elif isinstance(data, dict) and "polygons" in data:
        source_items = data["polygons"]
    elif isinstance(data, dict) and "shapes" in data:
        source_items = data["shapes"]
    elif isinstance(data, list):
        source_items = data
    else:
        raise ValueError(
            f"Unsupported page annotation structure in {annotation_path}; "
            "expected items, polygons, shapes, or a top-level list."
        )

    records: List[Dict[str, Any]] = []
    for source_index, item in enumerate(source_items):
        if not isinstance(item, dict):
            continue

        points = item.get("points") or item.get("polygon") or item.get("box")
        if points is None and "bbox" in item:
            x1, y1, x2, y2 = item["bbox"]
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if points is None:
            continue

        if item.get("shape_type") == "rectangle":
            polygon = _rectangle_to_polygon(points)
        else:
            polygon = _normalize_points(points)
        if len(polygon) < 3:
            continue

        records.append(
            {
                "index": len(records),
                "source_index": source_index,
                "box": polygon,
                "points": polygon,
                "text": extract_transcription(item),
            }
        )

    if not records:
        raise RuntimeError(f"No valid word polygons found in annotation: {annotation_path}")
    return records


def match_word_items(
    pred_items: Sequence[Mapping[str, Any]],
    gt_items: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.75,
) -> Dict[str, Any]:
    """Greedily match predicted boxes exactly as ``DetectionMetric`` does."""
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    gt_boxes = [polygon_to_bbox(item["box"]) for item in gt_items]
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: List[Dict[str, Any]] = []

    indexed_predictions = sorted(
        enumerate(pred_items),
        key=lambda indexed_item: int(
            indexed_item[1].get("detector_index", indexed_item[0])
        ),
    )
    for pred_index, pred in indexed_predictions:
        pred_box = polygon_to_bbox(pred["box"])
        best_iou = 0.0
        best_gt_index = -1

        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue
            iou = bbox_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index >= 0 and best_iou >= float(iou_threshold):
            matched_gt.add(best_gt_index)
            matched_pred.add(pred_index)
            prediction = str(pred.get("text", ""))
            reference = str(gt_items[best_gt_index].get("text", ""))
            matches.append(
                {
                    "pred_index": pred_index,
                    "gt_index": best_gt_index,
                    "iou": float(best_iou),
                    "prediction": prediction,
                    "reference": reference,
                    "text_exact": prediction == reference,
                }
            )

    unmatched_gt = [
        index for index in range(len(gt_items)) if index not in matched_gt
    ]
    unmatched_pred = [
        index for index in range(len(pred_items)) if index not in matched_pred
    ]
    return {
        "matches": matches,
        "unmatched_gt_indices": unmatched_gt,
        "unmatched_pred_indices": unmatched_pred,
        "tp": len(matches),
        "fp": len(unmatched_pred),
        "fn": len(unmatched_gt),
    }


def aggregate_edit_counts(
    prediction_reference_pairs: Iterable[Tuple[str, str]],
) -> Dict[str, Any]:
    """Aggregate strict S/D/I/N and CER over prediction/reference pairs."""
    substitutions = 0
    deletions = 0
    insertions = 0
    reference_characters = 0

    for prediction, reference in prediction_reference_pairs:
        sample_s, sample_d, sample_i = levenshtein_error_counts(
            prediction=str(prediction),
            reference=str(reference),
        )
        substitutions += sample_s
        deletions += sample_d
        insertions += sample_i
        reference_characters += len(str(reference))

    edit_errors = substitutions + deletions + insertions
    cer = (
        float(edit_errors / reference_characters)
        if reference_characters > 0
        else None
    )
    return {
        "substitutions": int(substitutions),
        "deletions": int(deletions),
        "insertions": int(insertions),
        "reference_characters": int(reference_characters),
        "edit_errors": int(edit_errors),
        "cer": cer,
    }


def build_e2e_character_pairs(
    pred_items: Sequence[Mapping[str, Any]],
    gt_items: Sequence[Mapping[str, Any]],
    matching: Mapping[str, Any],
) -> List[Tuple[str, str]]:
    """Build E2E edit pairs, penalizing missed GT and false-positive boxes."""
    pairs = [
        (str(match["prediction"]), str(match["reference"]))
        for match in matching["matches"]
    ]
    pairs.extend(
        ("", str(gt_items[index].get("text", "")))
        for index in matching["unmatched_gt_indices"]
    )
    pairs.extend(
        (str(pred_items[index].get("text", "")), "")
        for index in matching["unmatched_pred_indices"]
    )
    return pairs


def compute_e2e_recognition_metrics(
    pred_items: Sequence[Mapping[str, Any]],
    gt_items: Sequence[Mapping[str, Any]],
    matching: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compute strict E2E WA, CA, and edit-distance CER.

    Matched crops use exact transcription equality. Missed ground-truth words
    contribute deletions and false-positive boxes contribute insertions. No
    recognition tolerance is deducted.
    """
    substitutions = 0
    deletions = 0
    insertions = 0
    reference_characters = sum(len(str(item.get("text", ""))) for item in gt_items)
    exact_correct_words = 0
    matched_reference_characters = 0
    matched_raw_errors = 0

    for match in matching["matches"]:
        prediction = str(match["prediction"])
        reference = str(match["reference"])
        sample_s, sample_d, sample_i = levenshtein_error_counts(
            prediction=prediction,
            reference=reference,
        )
        sample_errors = sample_s + sample_d + sample_i
        substitutions += sample_s
        deletions += sample_d
        insertions += sample_i
        matched_reference_characters += len(reference)
        matched_raw_errors += sample_errors
        exact_correct_words += int(prediction == reference)

    for gt_index in matching["unmatched_gt_indices"]:
        missed_characters = len(str(gt_items[gt_index].get("text", "")))
        deletions += missed_characters

    for pred_index in matching["unmatched_pred_indices"]:
        false_positive_characters = len(
            str(pred_items[pred_index].get("text", ""))
        )
        insertions += false_positive_characters

    edit_errors = substitutions + deletions + insertions
    matched_count = len(matching["matches"])
    correct_reference_characters = (
        reference_characters - substitutions - deletions
    )
    cer = (
        float(edit_errors / reference_characters)
        if reference_characters > 0
        else None
    )
    word_accuracy = (
        float(exact_correct_words / len(gt_items)) if gt_items else 0.0
    )
    matched_word_accuracy = (
        float(exact_correct_words / matched_count) if matched_count else None
    )
    matched_cer = (
        float(matched_raw_errors / matched_reference_characters)
        if matched_reference_characters > 0
        else None
    )
    character_accuracy = (
        float(correct_reference_characters / reference_characters)
        if reference_characters > 0
        else None
    )

    return {
        "substitutions": int(substitutions),
        "deletions": int(deletions),
        "insertions": int(insertions),
        "reference_characters": int(reference_characters),
        "edit_errors": int(edit_errors),
        "strict_edit_errors": int(edit_errors),
        "cer": cer,
        "strict_cer": cer,
        "strict_exact_correct_words": int(exact_correct_words),
        "word_accuracy": word_accuracy,
        "strict_word_accuracy": word_accuracy,
        "matched_word_accuracy": matched_word_accuracy,
        "matched_strict_word_accuracy": matched_word_accuracy,
        "matched_cer": matched_cer,
        "matched_strict_cer": matched_cer,
        "matched_reference_characters": int(matched_reference_characters),
        "matched_strict_edit_errors": int(matched_raw_errors),
        "correct_reference_characters": int(correct_reference_characters),
        "strict_correct_reference_characters": int(correct_reference_characters),
        "character_accuracy": character_accuracy,
        "strict_character_accuracy": character_accuracy,
    }


def page_transcription_metrics(
    prediction: str,
    reference: str,
) -> Dict[str, Any]:
    """Compute page-level strict CER and normalized edit similarity."""
    counts = aggregate_edit_counts([(prediction, reference)])
    denominator = max(len(reference), len(prediction), 1)
    similarity = max(0.0, 1.0 - counts["edit_errors"] / denominator)
    return {
        **counts,
        "edit_similarity": float(similarity),
        "exact": bool(prediction == reference),
    }
