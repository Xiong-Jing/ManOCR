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
    parser.add_argument(
        "--detection-comparison",
        type=str,
        default="outputs/metrics/detection/detection_comparison_degraded_modelwise.json",
    )
    parser.add_argument("--recognition-summary", type=str, default="outputs/metrics/recognition/recognition_ablation_formal.json")
    parser.add_argument("--recognition-comparison", type=str, default="outputs/metrics/recognition/recognition_comparison.json")
    parser.add_argument("--full-ocr-summary", type=str, default="outputs/metrics/full_ocr/full_ocr_formal.json")
    parser.add_argument("--baseline-summary", type=str, default="outputs/metrics/baselines/external_baselines.json")
    parser.add_argument("--output", type=str, default="outputs/metrics/paper_tables.md")
    args = parser.parse_args()

    lines = ["# Paper Tables", ""]
    det_path = Path(args.detection_summary)
    det_comparison_path = Path(args.detection_comparison)
    rec_path = Path(args.recognition_summary)
    rec_comparison_path = Path(args.recognition_comparison)
    full_path = Path(args.full_ocr_summary)
    baseline_path = Path(args.baseline_summary)

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

    if det_comparison_path.exists():
        rows = load_json(det_comparison_path)
        lines.extend(
            [
                "## Detection Comparison",
                "",
                "| Model | Status | Test P | Test R | Test F |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            precision = row.get("precision")
            recall = row.get("recall")
            fmeasure = row.get("fmeasure")
            p_text = "-" if precision is None else f"{precision * 100:.2f}"
            r_text = "-" if recall is None else f"{recall * 100:.2f}"
            f_text = "-" if fmeasure is None else f"{fmeasure * 100:.2f}"
            lines.append(
                f"| {row['model']} | {row['status']} | {p_text} | {r_text} | {f_text} |"
            )
    else:
        lines.append(f"Detection comparison not found: `{det_comparison_path}`")

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

    if rec_comparison_path.exists():
        rows = load_json(rec_comparison_path)
        lines.extend(
            [
                "## Recognition Comparison",
                "",
                "| Model | Status | Test WA | Test CA | Test CER |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            wa = row.get("word_accuracy")
            ca = row.get("character_accuracy")
            cer = row.get("cer")
            wa_text = "-" if wa is None else f"{wa * 100:.2f}"
            ca_text = "-" if ca is None else f"{ca * 100:.2f}"
            cer_text = "-" if cer is None else f"{cer * 100:.2f}"
            lines.append(
                f"| {row['model']} | {row['status']} | {wa_text} | {ca_text} | {cer_text} |"
            )
    else:
        lines.append(f"Recognition comparison not found: `{rec_comparison_path}`")

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

    if baseline_path.exists():
        rows = load_json(baseline_path)
        lines.extend(
            [
                "## External Baselines",
                "",
                "| Model | Task | Status | F/WA | CER |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in rows:
            main_score = row.get("fmeasure")
            if main_score is None:
                main_score = row.get("word_accuracy")

            cer = row.get("cer")
            main_score_text = "-" if main_score is None else f"{main_score * 100:.2f}"
            cer_text = "-" if cer is None else f"{cer * 100:.2f}"

            lines.append(
                f"| {row['name']} | {row['task']} | {row['status']} "
                f"| {main_score_text} | {cer_text} |"
            )
    else:
        lines.append(f"External baseline summary not found: `{baseline_path}`")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
