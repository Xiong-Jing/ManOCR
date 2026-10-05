import argparse
import json
from pathlib import Path


EXPERIMENTS = [
    ("SVTR", "svtr_official_baseline"),
    ("SVTR + DAB", "svtr_official_dab"),
    ("SVTR + Lortho", "svtr_official_lortho"),
    ("SVTR + DAB + Lortho", "svtr_official_dab_lortho"),
]


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data["metrics"]


def fmt_percent(x: float) -> str:
    return f"{x * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/recognition")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []

    for model_name, exp_name in EXPERIMENTS:
        val_path = root / exp_name / "eval_val.json"
        test_path = root / exp_name / "eval_test.json"

        if args.allow_missing and (not val_path.exists() or not test_path.exists()):
            continue

        val = load_metrics(val_path)
        test = load_metrics(test_path)

        rows.append(
            {
                "model": model_name,
                "val_wa": val["word_accuracy"],
                "val_ca": val["character_accuracy"],
                "val_cer": val["cer"],
                "test_wa": test["word_accuracy"],
                "test_ca": test["character_accuracy"],
                "test_cer": test["cer"],
                "test_edit_distance": test["edit_distance"],
                "test_samples": test.get("num_samples", 0),
            }
        )

    out_dir = root
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "recognition_ablation_formal.md"
    json_path = out_dir / "recognition_ablation_formal.json"

    lines = []    
    lines.append("# Recognition Ablation Formal Results")
    lines.append("")
    lines.append("| Model | Val WA | Val CA | Val CER | Test WA | Test CA | Test CER | EditDist |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['model']} "
            f"| {fmt_percent(row['val_wa'])} "
            f"| {fmt_percent(row['val_ca'])} "
            f"| {fmt_percent(row['val_cer'])} "
            f"| {fmt_percent(row['test_wa'])} "
            f"| {fmt_percent(row['test_ca'])} "
            f"| {fmt_percent(row['test_cer'])} "
            f"| {row['test_edit_distance']:.4f} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))
    print(f"\nSaved markdown to: {md_path}")
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()
