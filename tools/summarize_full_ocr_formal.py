import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data["metrics"]


def fmt_percent(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/full_ocr")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        if args.allow_missing:
            root.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Missing full OCR metrics root: {root}")

    eval_files = sorted(root.glob("*/eval_full_ocr.json"))

    if (root / "eval_full_ocr.json").exists():
        eval_files.insert(0, root / "eval_full_ocr.json")

    if not eval_files:
        if args.allow_missing:
            print(f"No eval_full_ocr.json files found under: {root}")
            return
        raise FileNotFoundError(f"No eval_full_ocr.json files found under: {root}")

    rows = []

    for path in eval_files:
        metrics = load_metrics(path)
        name = path.parent.name if path.parent != root else "full_ocr"

        rows.append(
            {
                "experiment": name,
                "det_p": metrics["detection_precision"],
                "det_r": metrics["detection_recall"],
                "det_f": metrics["detection_fmeasure"],
                "e2e_wa": metrics["end_to_end_word_accuracy"],
                "matched_wa": metrics["matched_word_accuracy"],
                "matched_ca": metrics["matched_character_accuracy"],
                "matched_cer": metrics["matched_cer"],
                "text_eval_available": metrics.get("text_eval_available", metrics["matched_text_samples"] > 0),
                "matched_text_samples": metrics["matched_text_samples"],
            }
        )

    md_path = root / "full_ocr_formal.md"
    json_path = root / "full_ocr_formal.json"

    lines = []
    lines.append("# Full OCR Formal Results")
    lines.append("")
    lines.append("| Experiment | Det P | Det R | Det F | E2E WA | Matched WA | Matched CA | Matched CER | Text Eval | Text Samples |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")

    for row in rows:
        lines.append(
            f"| {row['experiment']} "
            f"| {fmt_percent(row['det_p'])} "
            f"| {fmt_percent(row['det_r'])} "
            f"| {fmt_percent(row['det_f'])} "
            f"| {fmt_percent(row['e2e_wa'])} "
            f"| {fmt_percent(row['matched_wa'])} "
            f"| {fmt_percent(row['matched_ca'])} "
            f"| {fmt_percent(row['matched_cer'])} "
            f"| {'yes' if row['text_eval_available'] else 'no'} "
            f"| {row['matched_text_samples']} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))
    print(f"\nSaved markdown to: {md_path}")
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()
