from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image

from manchu_ocr.metrics.e2e_ocr_metrics import (
    aggregate_edit_counts,
    build_e2e_character_pairs,
    compute_e2e_recognition_metrics,
    match_word_items,
    page_transcription_metrics,
    parse_page_annotation,
)
from manchu_ocr.pipelines.full_ocr_pipeline import FullOCRPipeline
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.experiment_result import (
    ExperimentTimer,
    build_experiment_result,
    collect_torch_runtime,
    format_experiment_result,
    split_key,
)
from manchu_ocr.utils.file_io import save_json
from manchu_ocr.utils.logger import setup_logger
from manchu_ocr.utils.seed import set_seed


STRICT_IOU_THRESHOLD = 0.75


def load_manifest(path: Path) -> List[Tuple[Path, Path]]:
    samples: List[Tuple[Path, Path]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid manifest line {line_index} in {path}: {line!r}"
                )
            samples.append((Path(parts[0]), Path(parts[1])))

    if not samples:
        raise RuntimeError(f"No samples found in manifest: {path}")
    return samples


def preflight_annotations(
    samples: Sequence[Tuple[Path, Path]],
) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """Read every annotation before loading either GPU model."""
    page_items: List[List[Dict[str, Any]]] = []
    missing_transcriptions: List[str] = []

    for image_path, annotation_path in samples:
        if not image_path.is_file():
            raise FileNotFoundError(f"Page image not found: {image_path}")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Page annotation not found: {annotation_path}")

        items = parse_page_annotation(annotation_path)
        page_items.append(items)
        for item in items:
            if not item["text"]:
                missing_transcriptions.append(
                    f"{annotation_path}#item[{item['source_index']}]"
                )

    return page_items, missing_transcriptions


def require_formal_transcriptions(missing_transcriptions: Sequence[str]) -> None:
    if not missing_transcriptions:
        return
    examples = "\n".join(f"  - {item}" for item in missing_transcriptions[:10])
    raise RuntimeError(
        "Formal E2E OCR evaluation requires a transcription for every ground-truth "
        "word, but "
        f"{len(missing_transcriptions)} word(s) have only an empty or generic "
        "detection label. No model inference was started. Add text/transcription "
        "fields to page annotations while keeping the existing split manifests. "
        "Examples:\n"
        f"{examples}"
    )


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def safe_fmeasure(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _compact_columns(columns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "column_index": column.get("column_index"),
            "text": column.get("text", ""),
            "item_indices": [
                item.get("index") for item in column.get("items", [])
            ],
        }
        for column in columns
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate detector-predicted word crops with the project's single "
            "strict detection and recognition protocol."
        )
    )
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--rec-config", required=True)
    parser.add_argument("--rec-checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="image<TAB>page_json")
    parser.add_argument("--pipeline-name", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", default="outputs/metrics/e2e_ocr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.04)
    parser.add_argument("--save-records", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate page files and word transcriptions without loading models.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run_timer = ExperimentTimer()
    args = _parse_args(argv)
    logger = setup_logger("eval_e2e_ocr")

    samples = load_manifest(Path(args.manifest))
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]

    gt_pages, missing_transcriptions = preflight_annotations(samples)
    require_formal_transcriptions(missing_transcriptions)
    if args.preflight_only:
        logger.info(
            f"E2E annotation preflight passed: pages={len(samples)}, "
            f"words={sum(len(items) for items in gt_pages)}"
        )
        return 0

    runtime_device = torch.device(args.device)
    if runtime_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(runtime_device)

    det_cfg = load_yaml(args.det_config)
    rec_cfg = load_yaml(args.rec_config)
    detector_name = str(det_cfg.get("experiment", {}).get("name", "detection"))
    recognizer_name = str(
        rec_cfg.get("experiment", {}).get("name", "recognition")
    )
    pipeline_name = args.pipeline_name or f"{detector_name}+{recognizer_name}"

    iou_threshold = STRICT_IOU_THRESHOLD
    evaluation_seed = int(det_cfg.get("train", {}).get("seed", 42))
    set_seed(evaluation_seed)

    pipeline = FullOCRPipeline(
        detection_config=args.det_config,
        detection_checkpoint=args.det_checkpoint,
        recognition_config=args.rec_config,
        recognition_checkpoint=args.rec_checkpoint,
        device=args.device,
        crop_padding=args.crop_padding,
        crop_padding_ratio=args.crop_padding_ratio,
    )
    logger.info(
        "Strict protocol: detector IoU=0.75, no evaluation degradation, "
        "exact-match E2E WA, CA=(N-S-D)/N, CER=(S+D+I)/N."
    )

    total_gt_words = 0
    total_pred_words = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_exact_words = 0
    e2e_character_pairs: List[Tuple[str, str]] = []
    matched_crop_pairs: List[Tuple[str, str]] = []
    page_text_pairs: List[Tuple[str, str]] = []
    page_similarities: List[float] = []
    exact_pages = 0
    records: List[Dict[str, Any]] = []

    for page_number, ((image_path, annotation_path), raw_gt_items) in enumerate(
        zip(samples, gt_pages),
        start=1,
    ):
        page_image = Image.open(image_path).convert("RGB")
        gt_columns = FullOCRPipeline.group_columns(raw_gt_items)
        gt_items = [
            item for column in gt_columns for item in column.get("items", [])
        ]
        gt_page_text = FullOCRPipeline.build_page_text(gt_columns)

        prediction = pipeline(page_image)
        if prediction.get("uses_ground_truth_crops") is not False:
            raise RuntimeError(
                "FullOCRPipeline did not affirm detector-only crop provenance."
            )
        pred_items = prediction["items"]
        matching = match_word_items(
            pred_items=pred_items,
            gt_items=gt_items,
            iou_threshold=iou_threshold,
        )

        recognition_metrics = compute_e2e_recognition_metrics(
            pred_items=pred_items,
            gt_items=gt_items,
            matching=matching,
        )
        exact_word_count = int(
            recognition_metrics["strict_exact_correct_words"]
        )
        character_pairs = build_e2e_character_pairs(
            pred_items=pred_items,
            gt_items=gt_items,
            matching=matching,
        )
        page_text = str(prediction.get("page_text", ""))
        page_metrics = page_transcription_metrics(page_text, gt_page_text)

        total_gt_words += len(gt_items)
        total_pred_words += len(pred_items)
        total_tp += int(matching["tp"])
        total_fp += int(matching["fp"])
        total_fn += int(matching["fn"])
        total_exact_words += exact_word_count
        e2e_character_pairs.extend(character_pairs)
        matched_crop_pairs.extend(
            (str(match["prediction"]), str(match["reference"]))
            for match in matching["matches"]
        )
        page_text_pairs.append((page_text, gt_page_text))
        page_similarities.append(float(page_metrics["edit_similarity"]))
        exact_pages += int(page_metrics["exact"])

        detailed_matches: List[Dict[str, Any]] = []
        for match in matching["matches"]:
            pred_item = pred_items[match["pred_index"]]
            gt_item = gt_items[match["gt_index"]]
            detailed_matches.append(
                {
                    **match,
                    "text_correct": bool(match["text_exact"]),
                    "pred_box": pred_item["box"],
                    "gt_box": gt_item["box"],
                    "det_score": float(pred_item.get("det_score", 0.0)),
                    "rec_confidence": float(
                        pred_item.get("rec_confidence", 0.0)
                    ),
                }
            )

        records.append(
            {
                "image_path": str(image_path).replace("\\", "/"),
                "annotation_path": str(annotation_path).replace("\\", "/"),
                "uses_ground_truth_crops": False,
                "crop_source": "detector_predictions",
                "reading_order": (
                    "columns_left_to_right_words_top_to_bottom"
                ),
                "detector_iou_threshold": float(iou_threshold),
                "evaluation_degradation_enabled": False,
                "num_gt_words": len(gt_items),
                "num_predicted_boxes": len(pred_items),
                "detection_tp": matching["tp"],
                "false_positive_boxes": matching["fp"],
                "missed_words": matching["fn"],
                "e2e_exact_words": exact_word_count,
                "e2e_word_accuracy": safe_ratio(
                    exact_word_count,
                    len(gt_items),
                ),
                "e2e_exact_word_accuracy": safe_ratio(
                    exact_word_count,
                    len(gt_items),
                ),
                "e2e_character_counts": aggregate_edit_counts(character_pairs),
                "e2e_recognition_metrics": recognition_metrics,
                "ground_truth_page_text": gt_page_text,
                "predicted_page_text": page_text,
                "page_transcription_metrics": page_metrics,
                "ground_truth_columns": _compact_columns(gt_columns),
                "predicted_columns": _compact_columns(
                    prediction.get("columns", [])
                ),
                "matches": detailed_matches,
                "missed_ground_truth_items": [
                    gt_items[index]
                    for index in matching["unmatched_gt_indices"]
                ],
                "false_positive_predictions": [
                    pred_items[index]
                    for index in matching["unmatched_pred_indices"]
                ],
                "pred_items": pred_items,
            }
        )

        logger.info(
            f"[{page_number}/{len(samples)}] {image_path.name} "
            f"GT={len(gt_items)} Pred={len(pred_items)} "
            f"TP={matching['tp']} FP={matching['fp']} FN={matching['fn']} "
            f"ExactCorrect={exact_word_count}"
        )

    precision = safe_ratio(total_tp, total_tp + total_fp)
    recall = safe_ratio(total_tp, total_tp + total_fn)
    fmeasure = safe_fmeasure(precision, recall)
    e2e_counts = aggregate_edit_counts(e2e_character_pairs)
    matched_crop_counts = aggregate_edit_counts(matched_crop_pairs)
    page_counts = aggregate_edit_counts(page_text_pairs)

    metrics = {
        "num_pages": int(len(samples)),
        "num_ground_truth_words": int(total_gt_words),
        "num_predicted_boxes": int(total_pred_words),
        "matching_method": "greedy_bbox_iou_same_as_detection_evaluator",
        "iou_threshold": float(iou_threshold),
        "iou_threshold_source": "project_strict_protocol",
        "degraded_evaluation": False,
        "evaluation_seed": int(evaluation_seed),
        "crop_source": "detector_predictions",
        "uses_ground_truth_crops": False,
        "reading_order": "columns_left_to_right_words_top_to_bottom",
        "detection_precision": float(precision),
        "detection_recall": float(recall),
        "detection_fmeasure": float(fmeasure),
        "detection_tp": int(total_tp),
        "detection_fp": int(total_fp),
        "detection_fn": int(total_fn),
        "missed_words": int(total_fn),
        "missed_word_rate": safe_ratio(total_fn, total_gt_words),
        "false_positive_boxes": int(total_fp),
        "false_positive_box_rate": safe_ratio(total_fp, total_pred_words),
        "e2e_exact_correct_words": int(total_exact_words),
        "e2e_word_accuracy": safe_ratio(total_exact_words, total_gt_words),
        "e2e_exact_word_accuracy": safe_ratio(
            total_exact_words,
            total_gt_words,
        ),
        "e2e_word_accuracy_definition": (
            "IoU-matched words with exactly correct transcription divided by "
            "all ground-truth words"
        ),
        "matched_word_accuracy": safe_ratio(total_exact_words, total_tp),
        "matched_exact_word_accuracy": safe_ratio(total_exact_words, total_tp),
        "e2e_substitutions": int(e2e_counts["substitutions"]),
        "e2e_deletions": int(e2e_counts["deletions"]),
        "e2e_insertions": int(e2e_counts["insertions"]),
        "e2e_reference_characters": int(e2e_counts["reference_characters"]),
        "e2e_edit_errors": int(e2e_counts["edit_errors"]),
        "e2e_strict_edit_errors": int(e2e_counts["edit_errors"]),
        "e2e_strict_cer": e2e_counts["cer"],
        "e2e_cer": e2e_counts["cer"],
        "e2e_cer_definition": "Strict corpus-level (S+D+I)/N",
        "e2e_character_accuracy": (
            float(
                (
                    int(e2e_counts["reference_characters"])
                    - int(e2e_counts["substitutions"])
                    - int(e2e_counts["deletions"])
                )
                / max(int(e2e_counts["reference_characters"]), 1)
            )
        ),
        "e2e_strict_character_accuracy": (
            float(
                (
                    int(e2e_counts["reference_characters"])
                    - int(e2e_counts["substitutions"])
                    - int(e2e_counts["deletions"])
                )
                / max(int(e2e_counts["reference_characters"]), 1)
            )
        ),
        "matched_crop_substitutions": int(
            matched_crop_counts["substitutions"]
        ),
        "matched_crop_deletions": int(matched_crop_counts["deletions"]),
        "matched_crop_insertions": int(matched_crop_counts["insertions"]),
        "matched_crop_reference_characters": int(
            matched_crop_counts["reference_characters"]
        ),
        "matched_crop_strict_cer": matched_crop_counts["cer"],
        "matched_crop_cer": matched_crop_counts["cer"],
        "matched_crop_cer_note": (
            "Strict recognizer-only audit on IoU-matched detector crops"
        ),
        "page_transcription_substitutions": int(page_counts["substitutions"]),
        "page_transcription_deletions": int(page_counts["deletions"]),
        "page_transcription_insertions": int(page_counts["insertions"]),
        "page_transcription_reference_characters": int(
            page_counts["reference_characters"]
        ),
        "page_transcription_cer": page_counts["cer"],
        "page_transcription_cer_is_strict": True,
        "page_transcription_score": float(
            sum(page_similarities) / max(len(page_similarities), 1)
        ),
        "page_exact_accuracy": safe_ratio(exact_pages, len(samples)),
        "page_text_separator_policy": (
            "single spaces between word tokens and newline between columns; "
            "both are included in page-level edit metrics"
        ),
        "recognition_tolerance_applied": False,
        "recognition_metric_protocol": "strict",
    }

    if runtime_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(runtime_device)

    timing = run_timer.finish()
    runtime_details = collect_torch_runtime(runtime_device)
    runtime_details.update(
        {
            "num_samples": int(len(samples)),
            "samples_per_second": round(
                float(len(samples)) / max(timing.elapsed_seconds, 1e-12),
                6,
            ),
        }
    )

    det_architecture = str(det_cfg.get("model", {}).get("name", "detection"))
    rec_architecture = str(rec_cfg.get("model", {}).get("name", "recognition"))
    result = build_experiment_result(
        task="e2e_ocr",
        experiment_name=f"e2e_ocr__{detector_name}__{recognizer_name}",
        model_name=pipeline_name,
        model_architecture=f"{det_architecture}+{rec_architecture}",
        split=args.split,
        config_path=(
            f"detection={args.det_config};recognition={args.rec_config}"
        ),
        checkpoint_path=(
            f"detection={args.det_checkpoint};recognition={args.rec_checkpoint}"
        ),
        manifest_path=args.manifest,
        metrics=metrics,
        timing=timing,
        runtime_details=runtime_details,
        extra_fields={
            "pipeline_name": pipeline_name,
            "det_config": args.det_config,
            "det_checkpoint": args.det_checkpoint,
            "rec_config": args.rec_config,
            "rec_checkpoint": args.rec_checkpoint,
            "checkpoints": {
                "detection": args.det_checkpoint,
                "recognition": args.rec_checkpoint,
            },
            "models": {
                "detection": detector_name,
                "recognition": recognizer_name,
            },
            "protocol": {
                "crop_source": "detector_predictions",
                "uses_ground_truth_crops": False,
                "matching": "greedy_bbox_iou",
                "iou_threshold": float(iou_threshold),
                "iou_threshold_source": "project_strict_protocol",
                "degraded_evaluation": False,
                "recognition_metrics": "strict_exact_wa_standard_ca_cer",
                "reading_order": (
                    "columns_left_to_right_words_top_to_bottom"
                ),
            },
        },
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"eval_{split_key(args.split)}.json"
    save_json(result, result_path)
    if args.save_records:
        save_json(
            records,
            output_dir / f"e2e_records_{split_key(args.split)}.json",
        )

    logger.info(
        f"E2E OCR finished. DetF={metrics['detection_fmeasure']:.4f}, "
        f"E2E_WA={metrics['e2e_word_accuracy']:.4f}, "
        f"E2E_CER={metrics['e2e_cer']:.4f}, "
        f"Page_CER={metrics['page_transcription_cer']:.4f}, "
        f"Missed={metrics['missed_words']}"
    )
    logger.info(format_experiment_result(result))
    logger.info(f"Saved metrics to: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
