import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENTS = [
    ("DBNet++", "dbnetpp_official_baseline"),
    ("+ VSAA", "dbnetpp_vsaa"),
    ("+ AS", "dbnetpp_asym_shrink"),
    ("+ VSAA + AS", "dbnetpp_vsaa_asym_shrink"),
]


def load_history(exp_name: str, root: Path):
    path = root / exp_name / "metrics.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric(
    metric_key: str,
    ylabel: str,
    output_name: str,
    root: Path,
    out_dir: Path,
    allow_missing: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for label, exp_name in EXPERIMENTS:
        history_path = root / exp_name / "metrics.json"

        if allow_missing and not history_path.exists():
            continue

        history = load_history(exp_name, root=root)
        epochs = [item["epoch"] for item in history]
        values = [item[metric_key] for item in history]

        if metric_key.startswith("val_") and metric_key not in ["val_loss"]:
            values = [v * 100 for v in values]

        plt.plot(epochs, values, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = out_dir / output_name
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[OK] saved {output_path}")


def plot_main_detection_curves(
    root: Path,
    out_dir: Path,
    exp_name: str = "dbnetpp_vsaa_asym_shrink",
    label: str = "DBNet++ + VSAA + AS",
    allow_missing: bool = False,
) -> None:
    history_path = root / exp_name / "metrics.json"

    if allow_missing and not history_path.exists():
        return

    history = load_history(exp_name, root=root)
    epochs = [item["epoch"] for item in history]

    main_dir = out_dir / "main_models"
    main_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train Loss")
    plt.plot(epochs, [item["val_loss"] for item in history], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(label)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path = main_dir / f"fig_main_detection_{exp_name}_loss.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] saved {output_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["val_precision"] * 100 for item in history], label="Precision")
    plt.plot(epochs, [item["val_recall"] * 100 for item in history], label="Recall")
    plt.plot(epochs, [item["val_fmeasure"] * 100 for item in history], label="F-measure")
    plt.xlabel("Epoch")
    plt.ylabel("Metric (%)")
    plt.title(label)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path = main_dir / f"fig_main_detection_{exp_name}_metrics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] saved {output_path}")

    component_keys = [
        ("train_prob_loss", "Train Prob"),
        ("train_binary_loss", "Train Binary"),
        ("train_thresh_loss", "Train Thresh"),
        ("val_prob_loss", "Val Prob"),
        ("val_binary_loss", "Val Binary"),
        ("val_thresh_loss", "Val Thresh"),
    ]
    if all(key in history[0] for key, _ in component_keys):
        plt.figure(figsize=(8, 5))
        for key, name in component_keys:
            plt.plot(epochs, [item[key] for item in history], label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{label} Loss Components")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        output_path = main_dir / f"fig_main_detection_{exp_name}_loss_components.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[OK] saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/detection")
    parser.add_argument("--out-dir", type=str, default="outputs/visualizations/paper_figures")
    parser.add_argument("--main-exp", type=str, default="dbnetpp_vsaa_asym_shrink")
    parser.add_argument("--main-label", type=str, default="DBNet++ + VSAA + AS")
    parser.add_argument("--no-main-plots", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)


    plot_metric("train_loss", "Training Loss", "fig_detection_train_loss.png", root, out_dir, allow_missing=args.allow_missing)
    plot_metric("val_loss", "Validation Loss", "fig_detection_val_loss.png", root, out_dir, allow_missing=args.allow_missing)
    plot_metric("val_precision", "Precision (%)", "fig_detection_precision.png", root, out_dir, allow_missing=args.allow_missing)
    plot_metric("val_recall", "Recall (%)", "fig_detection_recall.png", root, out_dir, allow_missing=args.allow_missing)
    plot_metric("val_fmeasure", "F-measure (%)", "fig_detection_fmeasure.png", root, out_dir, allow_missing=args.allow_missing)

    if not args.no_main_plots:
        plot_main_detection_curves(
            root=root,
            out_dir=out_dir,
            exp_name=args.main_exp,
            label=args.main_label,
            allow_missing=args.allow_missing,
        )


if __name__ == "__main__":
    main()
