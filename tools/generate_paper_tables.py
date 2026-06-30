import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detection-summary",
        type=str,
        default="outputs/metrics/detection/detection_ablation_degraded_modelwise.json",
    )
    parser.add_argument("--recognition-summary", type=str, default="outputs/metrics/recognition/recognition_ablation_formal.json")
    parser.add_argument("--full-ocr-summary", type=str, default="outputs/metrics/full_ocr/full_ocr_formal.json")
    parser.add_argument("--output", type=str, default="outputs/metrics/paper_tables.md")
    args = parser.parse_args()

    lines = ["# Paper Tables", ""]
    det_path = Path(args.detection_summary)
    rec_path = Path(args.recognition_summary)
    full_path = Path(args.full_ocr_summary)

    if det_path.exists():
        rows = load_json(det_path)
        lines.extend(
            [
                "## Detection Ablation",
                "",
                "| Model | Val F | Test F | TP | FP | FN |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['model']} | {row['val_f'] * 100:.2f} | {row['test_f'] * 100:.2f} "
                f"| {row['test_tp']} | {row['test_fp']} | {row['test_fn']} |"
            )
    else:
        lines.append(f"Detection summary not found: `{det_path}`")

    lines.append("")

    if rec_path.exists():
        rows = load_json(rec_path)
        lines.extend(
            [
                "## Recognition Ablation",
                "",
                "| Model | Val WA | Test WA | Test CER |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['model']} | {row['val_wa'] * 100:.2f} "
                f"| {row['test_wa'] * 100:.2f} | {row['test_cer'] * 100:.2f} |"
            )
    else:
        lines.append(f"Recognition summary not found: `{rec_path}`")

    lines.append("")

    if full_path.exists():
        rows = load_json(full_path)
        lines.extend(
            [
                "## Full OCR Inference",
                "",
                "| Experiment | Text Evaluation | JSON Output | Page Text Output |",
                "|---|---|---|---|",
            ]
        )
        for row in rows:
            text_eval_available = bool(row.get("text_eval_available", False))
            text_eval = "available" if text_eval_available else "not available"
            lines.append(
                f"| {row['experiment']} | {text_eval} "
                f"| outputs/predictions/full_ocr/full_page_predictions.json "
                f"| outputs/predictions/full_ocr/page_texts |"
            )
    else:
        lines.extend(
            [
                "## Full OCR Inference",
                "",
                "| JSON Output | Page Text Output |",
                "|---|---|",
                "| outputs/predictions/full_ocr/full_page_predictions.json | outputs/predictions/full_ocr/page_texts |",
            ]
        )

    lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
