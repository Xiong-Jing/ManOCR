import argparse
import json
from pathlib import Path


BASE_COMPARISONS = [
    {
        "model": "EAST",
        "source": "detection",
        "exp_name": "east_baseline",
    },
    {
        "model": "CRAFT",
        "source": "detection",
        "exp_name": "craft_baseline",
    },
    {
        "model": "DBNet",
        "source": "detection",
        "exp_name": "dbnet_baseline",
    },
    {
        "model": "PP-OCRv5 Det",
        "source": "detection",
        "exp_name": "ppocrv5_det_baseline",
    },
    {
        "model": "Hi-SAM",
        "source": "detection",
        "exp_name": "hisam_baseline",
    },
    {
        "model": "Ours (DBNet++ + VSAA + AS)",
        "source": "detection",
        "exp_name": "dbnetpp_vsaa_asym_shrink",
    },
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("metrics", data)


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection-root", type=str, default="outputs/metrics/detection")
    parser.add_argument("--baseline-root", type=str, default="outputs/metrics/baselines")
    parser.add_argument("--output-prefix", type=str, default="detection_comparison")
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Metric file suffix, e.g. _degraded for eval_test_degraded.json.",
    )
    args = parser.parse_args()

    detection_root = Path(args.detection_root)
    baseline_root = Path(args.baseline_root)
    suffix = args.suffix.strip()

    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    rows = []

    for item in BASE_COMPARISONS:
        root = baseline_root if item["source"] == "baseline" else detection_root
        result_path = root / item["exp_name"] / f"eval_test{suffix}.json"
        metrics = load_json(result_path)

        rows.append(
            {
                "model": item["model"],
                "source": item["source"],
                "status": "available" if metrics is not None else "missing",
                "result_path": str(result_path).replace("\\", "/"),
                "precision": None if metrics is None else metrics.get("precision"),
                "recall": None if metrics is None else metrics.get("recall"),
                "fmeasure": None if metrics is None else metrics.get("fmeasure"),
                "tp": None if metrics is None else metrics.get("tp"),
                "fp": None if metrics is None else metrics.get("fp"),
                "fn": None if metrics is None else metrics.get("fn"),
            }
        )

    out_dir = detection_root
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{args.output_prefix}.md"
    json_path = out_dir / f"{args.output_prefix}.json"

    lines = []
    lines.append("# Detection Comparison Results")
    lines.append("")
    lines.append("| Model | Status | P | R | F | TP | FP | FN |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['model']} "
            f"| {row['status']} "
            f"| {fmt_percent(row['precision'])} "
            f"| {fmt_percent(row['recall'])} "
            f"| {fmt_percent(row['fmeasure'])} "
            f"| {row['tp'] if row['tp'] is not None else '-'} "
            f"| {row['fp'] if row['fp'] is not None else '-'} "
            f"| {row['fn'] if row['fn'] is not None else '-'} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))
    print(f"\nSaved markdown to: {md_path}")
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()
