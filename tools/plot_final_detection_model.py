import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path: Path) -> dict:
    data = load_json(path)
    return data.get("metrics", data)


def normalize_suffix(suffix: str) -> str:
    suffix = suffix.strip()

    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    return suffix


def plot_train_loss(root: Path, out_dir: Path, exp_name: str, label: str) -> None:
    history = load_json(root / exp_name / "metrics.json")

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{label} Training Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = out_dir / f"fig_final_detection_{exp_name}_train_loss.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[OK] saved {output_path}")


def plot_val_test_metrics(
    root: Path,
    out_dir: Path,
    exp_name: str,
    label: str,
    suffix: str,
) -> None:
    val = load_metrics(root / exp_name / f"eval_val{suffix}.json")
    test = load_metrics(root / exp_name / f"eval_test{suffix}.json")

    names = ["Precision", "Recall", "F-measure"]
    val_values = [
        val["precision"] * 100,
        val["recall"] * 100,
        val["fmeasure"] * 100,
    ]
    test_values = [
        test["precision"] * 100,
        test["recall"] * 100,
        test["fmeasure"] * 100,
    ]

    x = list(range(len(names)))
    width = 0.36

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar([item - width / 2 for item in x], val_values, width=width, label="Val")
    plt.bar([item + width / 2 for item in x], test_values, width=width, label="Test")
    plt.xticks(x, names)
    plt.ylabel("Metric (%)")
    plt.title(f"{label} Val/Test Metrics")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = out_dir / f"fig_final_detection_{exp_name}_val_test_metrics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[OK] saved {output_path}")

    summary = {
        "experiment": exp_name,
        "suffix": suffix,
        "val": {
            "precision": val["precision"],
            "recall": val["recall"],
            "fmeasure": val["fmeasure"],
            "tp": val["tp"],
            "fp": val["fp"],
            "fn": val["fn"],
            "num_gt": val["num_gt"],
            "num_pred": val["num_pred"],
        },
        "test": {
            "precision": test["precision"],
            "recall": test["recall"],
            "fmeasure": test["fmeasure"],
            "tp": test["tp"],
            "fp": test["fp"],
            "fn": test["fn"],
            "num_gt": test["num_gt"],
            "num_pred": test["num_pred"],
        },
    }

    output_json = out_dir / f"final_detection_{exp_name}_val_test_summary.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/detection")
    parser.add_argument("--out-dir", type=str, default="outputs/visualizations/paper_figures/main_models")
    parser.add_argument("--exp-name", type=str, default="dbnetpp_vsaa_asym_shrink")
    parser.add_argument("--label", type=str, default="DBNet++ + VSAA + AS")
    parser.add_argument("--suffix", type=str, default="_strict")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    suffix = normalize_suffix(args.suffix)

    plot_train_loss(
        root=root,
        out_dir=out_dir,
        exp_name=args.exp_name,
        label=args.label,
    )
    plot_val_test_metrics(
        root=root,
        out_dir=out_dir,
        exp_name=args.exp_name,
        label=args.label,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()
