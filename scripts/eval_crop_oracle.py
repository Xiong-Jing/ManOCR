"""Paired predicted/GT crop experiment on identical manually annotated pages."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from manchu_ocr.data.page_splits import read_manifest
from manchu_ocr.data.orthography import sha256_file
from manchu_ocr.metrics.e2e_ocr_metrics import (parse_page_annotation, match_word_items,
                                             build_e2e_character_pairs, aggregate_edit_counts)
from manchu_ocr.metrics.recognition_metrics import compute_recognition_metrics
from manchu_ocr.pipelines.crop_rectify import crop_polygon_bbox
from manchu_ocr.pipelines.full_ocr_pipeline import FullOCRPipeline
from manchu_ocr.utils.file_io import save_json
from manchu_ocr.utils.experiment_result import ExperimentTimer, collect_torch_runtime


def evaluate_crop_pair(pipeline, image, gt_items):
    """Share recognizer object, transform, decoder and crop padding across inputs."""
    prediction = pipeline(image)
    if prediction.get("uses_ground_truth_crops") is not False:
        raise ValueError("Predicted condition must use detector-only crops")
    gt_items = FullOCRPipeline.sort_items_reading_order(gt_items)
    matching = match_word_items(prediction["items"], gt_items, 0.75)
    pred_pairs = build_e2e_character_pairs(prediction["items"], gt_items, matching)
    oracle = []
    for item in gt_items:
        crop = crop_polygon_bbox(image, item["box"], padding=pipeline.crop_padding,
                                 padding_ratio=pipeline.crop_padding_ratio)
        result = pipeline.recognizer(crop)
        oracle.append({"source_index": item.get("source_index"), "box": item["box"],
                       "reference": item["text"], "prediction": result["text"]})
    return {
        "ground_truth_words": len(gt_items),
        "predicted_exact_words": sum(m["text_exact"] for m in matching["matches"]),
        "predicted_character_pairs": pred_pairs,
        "matching": matching, "oracle_items": oracle, "predicted_items": prediction["items"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="image<TAB>manually re-annotated page JSON")
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--rec-config", required=True)
    parser.add_argument("--rec-checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.04)
    parser.add_argument("--output-dir", default="outputs/metrics/crop_oracle")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    samples = read_manifest(args.manifest)
    if not samples or len({image for image, _ in samples}) != len(samples):
        raise ValueError("Need nonempty, unique page manifest")
    pages = []
    for image, annotation in samples:
        if not Path(image).is_file():
            raise FileNotFoundError(image)
        items = parse_page_annotation(annotation)
        if not items or any(not item["text"] for item in items):
            raise ValueError(f"Incomplete GT word annotations: {annotation}")
        pages.append(items)
    if args.preflight_only:
        print(f"OK: {len(pages)} pages, {sum(map(len, pages))} GT words")
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly use --device cpu for local checks")
    timer = ExperimentTimer()
    pipeline = FullOCRPipeline(args.det_config, args.det_checkpoint, args.rec_config, args.rec_checkpoint,
                               device=args.device, crop_padding=args.crop_padding,
                               crop_padding_ratio=args.crop_padding_ratio)
    records, pred_pairs, references, oracle_predictions = [], [], [], []
    exact = n_words = 0
    for (image_path, annotation_path), items in zip(samples, pages):
        with Image.open(image_path) as image:
            record = evaluate_crop_pair(pipeline, image.convert("RGB"), items)
        pred_pairs.extend(record.pop("predicted_character_pairs"))
        exact += record["predicted_exact_words"]
        n_words += record["ground_truth_words"]
        references.extend(item["reference"] for item in record["oracle_items"])
        oracle_predictions.extend(item["prediction"] for item in record["oracle_items"])
        records.append(dict(record, image_path=image_path, annotation_path=annotation_path,
                            image_sha256=sha256_file(image_path), annotation_sha256=sha256_file(annotation_path)))
    gt = compute_recognition_metrics(oracle_predictions, references)
    predicted = aggregate_edit_counts(pred_pairs)
    predicted["word_accuracy"] = exact / n_words
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    result = {
        "schema": "manchu.crop_oracle.v1", "split": args.split, "num_pages": len(samples),
        "manifest": args.manifest, "manifest_sha256": sha256_file(args.manifest),
        "detector_checkpoint_sha256": sha256_file(args.det_checkpoint),
        "recognizer_checkpoint_sha256": sha256_file(args.rec_checkpoint),
        "detector_config_sha256": sha256_file(args.det_config),
        "recognizer_config_sha256": sha256_file(args.rec_config),
        "protocol": {"iou": 0.75, "same_recognizer": True, "same_pages": True,
                     "crop_padding": args.crop_padding, "crop_padding_ratio": args.crop_padding_ratio,
                     "predicted_cer_includes_misses_and_false_positives": True,
                     "oracle_uses_ground_truth_crops": True},
        "detector_predicted_crops": predicted, "ground_truth_crops": gt,
        "delta_WA_percentage_points": 100 * (gt["word_accuracy"] - predicted["word_accuracy"]),
        "delta_CER_percentage_points": 100 * (gt["cer"] - predicted["cer"]),
        "runtime_seconds": timer.finish().elapsed_seconds, "runtime": collect_torch_runtime(args.device),
    }
    output = Path(args.output_dir)
    save_json(result, output / f"eval_{args.split}.json")
    save_json(records, output / f"records_{args.split}.json")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
