import argparse
import json
from pathlib import Path


COMPARISONS = [
    {
        "model": "CRNN",
        "path": "crnn_baseline/eval_test.json",
    },
    {
        "model": "PARSeq",
        "path": "parseq_baseline/eval_test.json",
    },
    {
        "model": "ABINet",
        "path": "abinet_baseline/eval_test.json",
    },
    {
        "model": "SVTRv2",
        "path": "svtrv2_baseline/eval_test.json",
    },
    {
        "model": "DCM",
        "path": "dcm_baseline/eval_test.json",
    },
    {
        "model": "Ours (SVTR + DAB + Lortho)",
        "path": "svtr_official_dab_lortho/eval_test.json",
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


def fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/recognition")
    parser.add_argument("--output-prefix", type=str, default="recognition_comparison")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []

    for item in COMPARISONS:
        result_path = root / item["path"]
        metrics = load_json(result_path)

        rows.append(
            {
                "model": item["model"],
                "status": "available" if metrics is not None else "missing",
                "result_path": str(result_path).replace("\\", "/"),
                "word_accuracy": None if metrics is None else metrics.get("word_accuracy"),
                "character_accuracy": None if metrics is None else metrics.get("character_accuracy"),
                "cer": None if metrics is None else metrics.get("cer"),
                "edit_distance": None if metrics is None else metrics.get("edit_distance"),
                "num_samples": None if metrics is None else metrics.get("num_samples"),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    md_path = root / f"{args.output_prefix}.md"
    json_path = root / f"{args.output_prefix}.json"

    lines = []
    lines.append("# Recognition Comparison Results")
    lines.append("")
    lines.append("| Model | Status | WA | CA | CER | EditDist | Samples |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['model']} "
            f"| {row['status']} "
            f"| {fmt_percent(row['word_accuracy'])} "
            f"| {fmt_percent(row['character_accuracy'])} "
            f"| {fmt_percent(row['cer'])} "
            f"| {fmt_float(row['edit_distance'])} "
            f"| {row['num_samples'] if row['num_samples'] is not None else '-'} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))
    print(f"\nSaved markdown to: {md_path}")
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()
