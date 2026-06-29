import argparse
import json
from pathlib import Path


EXPERIMENTS = [
    ("DBNet++", "dbnetpp_official_baseline"),
    ("DBNet++ + VSAA", "dbnetpp_vsaa"),
    ("DBNet++ + AS", "dbnetpp_asym_shrink"),
    ("DBNet++ + VSAA + AS", "dbnetpp_vsaa_asym_shrink"),
]


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data["metrics"]


def fmt(x: float) -> str:
    return f"{x * 100:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/detection")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional eval filename suffix, e.g. _iou70.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="detection_ablation_formal",
        help="Output filename prefix without extension.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only summarize eval_test*.json. Useful for degraded strict test reports.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    suffix = args.suffix.strip()

    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    rows = []

    for model_name, exp_name in EXPERIMENTS:
        val_path = root / exp_name / f"eval_val{suffix}.json"
        test_path = root / exp_name / f"eval_test{suffix}.json"

        if args.test_only:
            if args.allow_missing and not test_path.exists():
                continue
            test = load_metrics(test_path)
            rows.append(
                {
                    "model": model_name,
                    "test_p": test["precision"],
                    "test_r": test["recall"],
                    "test_f": test["fmeasure"],
                    "test_tp": test["tp"],
                    "test_fp": test["fp"],
                    "test_fn": test["fn"],
                    "test_gt": test["num_gt"],
                    "test_pred": test["num_pred"],
                }
            )
            continue

        val = load_metrics(val_path)
        test = load_metrics(test_path)

        rows.append(
            {
                "model": model_name,
                "val_p": val["precision"],
                "val_r": val["recall"],
                "val_f": val["fmeasure"],
                "test_p": test["precision"],
                "test_r": test["recall"],
                "test_f": test["fmeasure"],
                "test_tp": test["tp"],
                "test_fp": test["fp"],
                "test_fn": test["fn"],
                "test_gt": test["num_gt"],
                "test_pred": test["num_pred"],
            }
        )

    out_dir = root
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{args.output_prefix}.md"
    json_path = out_dir / f"{args.output_prefix}.json"

    lines = []
    title_suffix = f" ({suffix.lstrip('_')})" if suffix else ""
    lines.append(f"# Detection Ablation Formal Results{title_suffix}")
    lines.append("")
    if args.test_only:
        lines.append("| Model | Test P | Test R | Test F | TP | FP | FN |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| Model | Val P | Val R | Val F | Test P | Test R | Test F | TP | FP | FN |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        if args.test_only:
            lines.append(
                f"| {row['model']} "
                f"| {fmt(row['test_p'])} "
                f"| {fmt(row['test_r'])} "
                f"| {fmt(row['test_f'])} "
                f"| {row['test_tp']} "
                f"| {row['test_fp']} "
                f"| {row['test_fn']} |"
            )
        else:
            lines.append(
                f"| {row['model']} "
                f"| {fmt(row['val_p'])} "
                f"| {fmt(row['val_r'])} "
                f"| {fmt(row['val_f'])} "
                f"| {fmt(row['test_p'])} "
                f"| {fmt(row['test_r'])} "
                f"| {fmt(row['test_f'])} "
                f"| {row['test_tp']} "
                f"| {row['test_fp']} "
                f"| {row['test_fn']} |"
            )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))
    print(f"\nSaved markdown to: {md_path}")
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()
