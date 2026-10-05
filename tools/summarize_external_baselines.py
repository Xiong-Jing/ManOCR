import argparse
import json
from pathlib import Path

from manchu_ocr.utils.config import load_yaml


BASELINE_CONFIGS = [
    "configs/baselines/craft.yaml",
    "configs/baselines/east.yaml",
    "configs/baselines/dbnet.yaml",
    "configs/baselines/ppocrv5_det.yaml",
    "configs/baselines/hisam.yaml",
    "configs/baselines/crnn.yaml",
    "configs/baselines/parseq.yaml",
    "configs/baselines/abinet.yaml",
    "configs/baselines/svtrv2.yaml",
    "configs/baselines/svtrv2_nrtr.yaml",
    "configs/baselines/dcm.yaml",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline result: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("metrics", data)


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def resolve_result_path(result_path: Path, baseline_root: Path) -> Path:
    if result_path.is_absolute():
        return result_path

    parts = result_path.parts
    output_root = baseline_root.parent.parent

    if len(parts) >= 3 and parts[0] == "outputs" and parts[1] == "metrics":
        return output_root / Path(*parts[1:])

    if len(parts) == 1:
        return baseline_root / result_path.name

    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/baselines")
    args = parser.parse_args()

    out_dir = Path(args.root)
    rows = []

    for cfg_path in BASELINE_CONFIGS:
        cfg = load_yaml(cfg_path)
        baseline = cfg["baseline"]
        result_path = resolve_result_path(
            Path(cfg["expected_metrics"]["output_json"]),
            baseline_root=out_dir,
        )

        if result_path.exists():
            metrics = load_json(result_path)
            status = "available"
        else:
            metrics = {}
            status = "missing"

        rows.append(
            {
                "name": baseline["name"],
                "task": baseline["task"],
                "status": status,
                "result_path": str(result_path).replace("\\", "/"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "fmeasure": metrics.get("fmeasure"),
                "word_accuracy": metrics.get("word_accuracy"),
                "character_accuracy": metrics.get("character_accuracy"),
                "cer": metrics.get("cer"),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "external_baselines.md"
    json_path = out_dir / "external_baselines.json"

    lines = []
    lines.append("# External Baselines")
    lines.append("")
    lines.append("| Name | Task | Status | P | R | F | WA | CA | CER |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['name']} "
            f"| {row['task']} "
            f"| {row['status']} "
            f"| {fmt_percent(row['precision'])} "
            f"| {fmt_percent(row['recall'])} "
            f"| {fmt_percent(row['fmeasure'])} "
            f"| {fmt_percent(row['word_accuracy'])} "
            f"| {fmt_percent(row['character_accuracy'])} "
            f"| {fmt_percent(row['cer'])} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))
    print(f"\nSaved markdown to: {md_path}")
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()
