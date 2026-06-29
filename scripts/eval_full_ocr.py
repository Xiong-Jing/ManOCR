import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from manchu_ocr.metrics.detection_metrics import bbox_iou, polygon_to_bbox
from manchu_ocr.metrics.recognition_metrics import (
    compute_recognition_metrics,
    is_relaxed_word_correct,
    levenshtein_distance,
)
from manchu_ocr.pipelines.full_ocr_pipeline import FullOCRPipeline
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import save_json
from manchu_ocr.utils.logger import setup_logger


GENERIC_LABELS = {"", "text", "word", "manchu", "ignore"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Path) -> List[Tuple[Path, Path]]:
    items = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(f"Invalid manifest line {line_idx}: {repr(line)}")

            items.append((Path(parts[0]), Path(parts[1])))

    if not items:
        raise RuntimeError(f"No samples found in manifest: {path}")

    return items


def normalize_points(points: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points]


def rectangle_to_polygon(points: Sequence[Sequence[float]]) -> List[List[float]]:
    if len(points) != 2:
        raise ValueError(f"Rectangle should contain 2 points, got {len(points)}")

    x1, y1 = points[0]
    x2, y2 = points[1]
    left = min(float(x1), float(x2))
    right = max(float(x1), float(x2))
    top = min(float(y1), float(y2))
    bottom = max(float(y1), float(y2))

    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def extract_text(item: Dict) -> str:
    for key in ["text", "transcription", "label", "value"]:
        value = str(item.get(key, "")).strip()

        if value and value.lower() not in GENERIC_LABELS:
            return value

    return ""


def parse_gt_annotation(path: Path) -> List[Dict]:
    data = load_json(path)
    records = []

    if isinstance(data, dict) and "items" in data:
        source_items = data["items"]
    elif isinstance(data, dict) and "polygons" in data:
        source_items = data["polygons"]
    elif isinstance(data, dict) and "shapes" in data:
        source_items = data["shapes"]
    elif isinstance(data, list):
        source_items = data
    else:
        source_items = []

    for idx, item in enumerate(source_items):
        if not isinstance(item, dict):
            continue

        points = item.get("points") or item.get("polygon") or item.get("box")

        if points is None and "bbox" in item:
            x1, y1, x2, y2 = item["bbox"]
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        if points is None:
            continue

        if item.get("shape_type") == "rectangle":
            points = rectangle_to_polygon(points)
        else:
            points = normalize_points(points)

        if len(points) < 3:
            continue

        records.append(
            {
                "index": idx,
                "box": points,
                "text": extract_text(item),
            }
        )

    records = sorted(records, key=lambda item: (polygon_to_bbox(item["box"])[0], polygon_to_bbox(item["box"])[1]))
    return records


def match_predictions(
    pred_items: List[Dict],
    gt_items: List[Dict],
    iou_threshold: float,
    word_correct_max_edit_distance: int = 0,
    word_correct_require_first_char_match: bool = False,
    word_correct_require_second_char_match: bool = False,
    word_correct_require_last_char_match: bool = False,
) -> Tuple[List[Dict], Dict[str, int]]:
    matched_gt = set()
    matches = []
    fp = 0

    gt_boxes = [polygon_to_bbox(item["box"]) for item in gt_items]

    for pred_idx, pred in enumerate(pred_items):
        pred_box = polygon_to_bbox(pred["box"])
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = bbox_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            gt = gt_items[best_gt_idx]
            pred_text = str(pred.get("text", ""))
            gt_text = str(gt.get("text", ""))

            matches.append(
                {
                    "pred_index": pred_idx,
                    "gt_index": best_gt_idx,
                    "iou": float(best_iou),
                    "pred_text": pred_text,
                    "gt_text": gt_text,
                    "text_correct": bool(
                        gt_text
                        and is_relaxed_word_correct(
                            pred_text,
                            gt_text,
                            max_edit_distance=word_correct_max_edit_distance,
                            require_first_char_match=word_correct_require_first_char_match,
                            require_second_char_match=word_correct_require_second_char_match,
                            require_last_char_match=word_correct_require_last_char_match,
                        )
                    ),
                    "pred_box": pred["box"],
                    "gt_box": gt["box"],
                    "det_score": float(pred.get("det_score", 0.0)),
                    "rec_confidence": float(pred.get("rec_confidence", 0.0)),
                }
            )
        else:
            fp += 1

    fn = len(gt_items) - len(matched_gt)

    return matches, {"tp": len(matches), "fp": fp, "fn": fn}


def safe_fmeasure(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-config", type=str, required=True)
    parser.add_argument("--det-checkpoint", type=str, required=True)
    parser.add_argument("--rec-config", type=str, required=True)
    parser.add_argument("--rec-checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True, help="image_path<TAB>page_annotation_json")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics/full_ocr")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.04)
    parser.add_argument("--save-records", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("eval_full_ocr")
    samples = load_manifest(Path(args.manifest))

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    pipeline = FullOCRPipeline(
        detection_config=args.det_config,
        detection_checkpoint=args.det_checkpoint,
        recognition_config=args.rec_config,
        recognition_checkpoint=args.rec_checkpoint,
        device=args.device,
        crop_padding=args.crop_padding,
        crop_padding_ratio=args.crop_padding_ratio,
    )
    rec_cfg = load_yaml(args.rec_config)
    word_accuracy_edit_distance = int(
        rec_cfg.get("metrics", {}).get("word_accuracy_edit_distance", 0)
    )
    character_accuracy_edit_distance = int(
        rec_cfg.get("metrics", {}).get("character_accuracy_edit_distance", 0)
    )
    word_accuracy_require_first_char_match = bool(
        rec_cfg.get("metrics", {}).get("word_accuracy_require_first_char_match", False)
    )
    word_accuracy_require_second_char_match = bool(
        rec_cfg.get("metrics", {}).get("word_accuracy_require_second_char_match", False)
    )
    word_accuracy_require_last_char_match = bool(
        rec_cfg.get("metrics", {}).get("word_accuracy_require_last_char_match", False)
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_text = 0
    text_preds = []
    text_labels = []
    records = []

    for image_idx, (image_path, ann_path) in enumerate(samples, start=1):
        gt_items = parse_gt_annotation(ann_path)
        total_gt_text += sum(1 for item in gt_items if item.get("text"))
        pred = pipeline(image_path)
        pred_items = pred["items"]

        matches, det_stats = match_predictions(
            pred_items=pred_items,
            gt_items=gt_items,
            iou_threshold=args.iou_thresh,
            word_correct_max_edit_distance=word_accuracy_edit_distance,
            word_correct_require_first_char_match=word_accuracy_require_first_char_match,
            word_correct_require_second_char_match=word_accuracy_require_second_char_match,
            word_correct_require_last_char_match=word_accuracy_require_last_char_match,
        )

        total_tp += det_stats["tp"]
        total_fp += det_stats["fp"]
        total_fn += det_stats["fn"]

        for match in matches:
            if match["gt_text"]:
                text_preds.append(match["pred_text"])
                text_labels.append(match["gt_text"])

        records.append(
            {
                "image_path": str(image_path).replace("\\", "/"),
                "annotation_path": str(ann_path).replace("\\", "/"),
                "num_gt": len(gt_items),
                "num_pred": len(pred_items),
                **det_stats,
                "page_text": pred.get("page_text", ""),
                "flat_text": pred.get("flat_text", ""),
                "columns": [
                    {
                        "column_index": column.get("column_index"),
                        "text": column.get("text", ""),
                        "item_indices": [item.get("index") for item in column.get("items", [])],
                    }
                    for column in pred.get("columns", [])
                ],
                "matches": matches,
                "pred_items": pred_items,
            }
        )

        logger.info(
            f"[{image_idx}/{len(samples)}] {image_path.name} "
            f"GT={len(gt_items)} Pred={len(pred_items)} "
            f"TP={det_stats['tp']} FP={det_stats['fp']} FN={det_stats['fn']}"
        )

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    fmeasure = safe_fmeasure(precision, recall)

    if text_labels:
        rec_metrics = compute_recognition_metrics(
            text_preds,
            text_labels,
            word_correct_max_edit_distance=word_accuracy_edit_distance,
            char_correct_max_edit_distance=character_accuracy_edit_distance,
            word_correct_require_first_char_match=word_accuracy_require_first_char_match,
            word_correct_require_second_char_match=word_accuracy_require_second_char_match,
            word_correct_require_last_char_match=word_accuracy_require_last_char_match,
        )
        e2e_word_accuracy = (
            sum(
                is_relaxed_word_correct(
                    p,
                    g,
                    max_edit_distance=word_accuracy_edit_distance,
                    require_first_char_match=word_accuracy_require_first_char_match,
                    require_second_char_match=word_accuracy_require_second_char_match,
                    require_last_char_match=word_accuracy_require_last_char_match,
                )
                for p, g in zip(text_preds, text_labels)
            )
            / max(len(text_labels), 1)
        )
    else:
        rec_metrics = {
            "word_accuracy": None,
            "exact_word_accuracy": None,
            "character_accuracy": None,
            "cer": None,
            "strict_character_accuracy": None,
            "strict_cer": None,
            "edit_distance": None,
            "strict_edit_distance": None,
            "word_accuracy_edit_distance": word_accuracy_edit_distance,
            "character_accuracy_edit_distance": character_accuracy_edit_distance,
            "word_accuracy_require_first_char_match": word_accuracy_require_first_char_match,
            "word_accuracy_require_second_char_match": word_accuracy_require_second_char_match,
            "word_accuracy_require_last_char_match": word_accuracy_require_last_char_match,
        }
        e2e_word_accuracy = None

    metrics = {
        "num_images": len(samples),
        "iou_threshold": args.iou_thresh,
        "detection_precision": float(precision),
        "detection_recall": float(recall),
        "detection_fmeasure": float(fmeasure),
        "detection_tp": int(total_tp),
        "detection_fp": int(total_fp),
        "detection_fn": int(total_fn),
        "gt_text_items": int(total_gt_text),
        "matched_text_samples": int(len(text_labels)),
        "text_eval_available": bool(len(text_labels) > 0),
        "text_eval_note": "" if text_labels else "No non-generic text labels found in page annotations.",
        "end_to_end_word_accuracy": None if e2e_word_accuracy is None else float(e2e_word_accuracy),
        "matched_word_accuracy": None if rec_metrics["word_accuracy"] is None else float(rec_metrics["word_accuracy"]),
        "matched_exact_word_accuracy": None if rec_metrics.get("exact_word_accuracy") is None else float(rec_metrics["exact_word_accuracy"]),
        "matched_character_accuracy": None if rec_metrics["character_accuracy"] is None else float(rec_metrics["character_accuracy"]),
        "matched_cer": None if rec_metrics["cer"] is None else float(rec_metrics["cer"]),
        "matched_strict_character_accuracy": None if rec_metrics.get("strict_character_accuracy") is None else float(rec_metrics["strict_character_accuracy"]),
        "matched_strict_cer": None if rec_metrics.get("strict_cer") is None else float(rec_metrics["strict_cer"]),
        "matched_edit_distance": None if rec_metrics["edit_distance"] is None else float(rec_metrics["edit_distance"]),
        "word_accuracy_edit_distance": int(word_accuracy_edit_distance),
        "character_accuracy_edit_distance": int(character_accuracy_edit_distance),
        "word_accuracy_require_first_char_match": bool(word_accuracy_require_first_char_match),
        "word_accuracy_require_second_char_match": bool(word_accuracy_require_second_char_match),
        "word_accuracy_require_last_char_match": bool(word_accuracy_require_last_char_match),
    }

    result = {
        "det_config": args.det_config,
        "det_checkpoint": args.det_checkpoint,
        "rec_config": args.rec_config,
        "rec_checkpoint": args.rec_checkpoint,
        "manifest": args.manifest,
        "metrics": metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, output_dir / "eval_full_ocr.json")

    if args.save_records:
        save_json(records, output_dir / "full_ocr_records.json")

    e2e_text = "N/A" if metrics["end_to_end_word_accuracy"] is None else f"{metrics['end_to_end_word_accuracy']:.4f}"
    matched_cer_text = "N/A" if metrics["matched_cer"] is None else f"{metrics['matched_cer']:.4f}"

    logger.info(
        f"Full OCR evaluation finished. "
        f"DetF={metrics['detection_fmeasure']:.4f}, "
        f"E2E_WA={e2e_text}, "
        f"Matched_CER={matched_cer_text}"
    )
    logger.info(f"Saved metrics to: {output_dir / 'eval_full_ocr.json'}")


if __name__ == "__main__":
    main()
