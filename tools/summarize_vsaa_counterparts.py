from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENTS = [
    ("No attention baseline", "dbnetpp_official_baseline"),
    ("Strip Pooling counterpart", "dbnetpp_strip_pooling"),
    ("Coordinate Attention counterpart", "dbnetpp_coordinate_attention"),
    ("Horizontal-only strip", "dbnetpp_horizontal_strip"),
    ("Vertical-only strip", "dbnetpp_vertical_strip"),
    ("VSAA", "dbnetpp_vsaa"),
]


def load_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation result: {path}")

    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    required = {
        "model_name",
        "split",
        "checkpoint_path",
        "manifest_path",
        "metrics",
        "runtime",
        "runtime_seconds",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise KeyError(f"{path} is missing required result fields: {missing}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the six controlled VSAA counterpart experiments."
    )
    parser.add_argument("--root", default="outputs/metrics/detection")
    parser.add_argument("--suffix", default="_vsaa_counterpart")
    parser.add_argument("--output-prefix", default="detection_vsaa_counterparts")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    suffix = str(args.suffix).strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    rows: list[dict[str, Any]] = []
    for label, experiment_name in EXPERIMENTS:
        paths = {
            split: root / experiment_name / f"eval_{split}{suffix}.json"
            for split in ("val", "test")
        }
        missing_paths = [path for path in paths.values() if not path.is_file()]
        if missing_paths and args.allow_missing:
            continue
        if missing_paths:
            raise FileNotFoundError(
                "Missing counterpart result(s): "
                + ", ".join(str(path) for path in missing_paths)
            )

        validation = load_result(paths["val"])
        test = load_result(paths["test"])
        val_metrics = validation["metrics"]
        test_metrics = test["metrics"]
        postprocess = test.get("postprocess", {})
        evaluation_protocol = test.get("evaluation_protocol", {})

        rows.append(
            {
                "label": label,
                "experiment": experiment_name,
                "direction_module": test.get("direction_module"),
                "checkpoint_path": test["checkpoint_path"],
                "validation_manifest": validation["manifest_path"],
                "test_manifest": test["manifest_path"],
                "iou_threshold": postprocess.get("iou_thresh"),
                "strict_eval": bool(evaluation_protocol.get("strict", False)),
                "trainable_parameters": int(
                    test.get("runtime", {}).get("trainable_parameters", 0)
                ),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_fmeasure": float(val_metrics["fmeasure"]),
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "test_fmeasure": float(test_metrics["fmeasure"]),
                "validation_runtime_seconds": float(validation["runtime_seconds"]),
                "test_runtime_seconds": float(test["runtime_seconds"]),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{args.output_prefix}.json"
    csv_path = root / f"{args.output_prefix}.csv"
    markdown_path = root / f"{args.output_prefix}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    lines = [
        "# VSAA Counterpart Ablation Results",
        "",
        "| Variant | Params (M) | IoU | Strict | Val P | Val R | Val F | Test P | Test R | Test F |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} "
            f"| {row['trainable_parameters'] / 1_000_000.0:.3f} "
            f"| {float(row['iou_threshold']):.2f} "
            f"| {'yes' if row['strict_eval'] else 'no'} "
            f"| {100.0 * row['val_precision']:.2f} "
            f"| {100.0 * row['val_recall']:.2f} "
            f"| {100.0 * row['val_fmeasure']:.2f} "
            f"| {100.0 * row['test_precision']:.2f} "
            f"| {100.0 * row['test_recall']:.2f} "
            f"| {100.0 * row['test_fmeasure']:.2f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
