import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENTS = [
    ("SVTR", "svtr_official_baseline"),
    ("+ DAB", "svtr_official_dab"),
    ("+ Lortho", "svtr_official_lortho"),
    ("+ DAB + Lortho", "svtr_official_dab_lortho"),
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
    percent: bool = False,
    allow_missing: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for label, exp_name in EXPERIMENTS:
        history_path = root / exp_name / "metrics.json"

        if allow_missing and not history_path.exists():
            continue

        history = load_history(exp_name, root=root)
        epochs = [item["epoch"] for item in history]
        values = [item[metric_key] for item in history]

        if percent:
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


def plot_main_recognition_curves(
    root: Path,
    out_dir: Path,
    exp_name: str = "svtr_official_dab_lortho",
    label: str = "SVTR + DAB + Lortho",
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
    output_path = main_dir / f"fig_main_recognition_{exp_name}_loss.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] saved {output_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["val_word_accuracy"] * 100 for item in history], label="WA")
    plt.plot(epochs, [item["val_character_accuracy"] * 100 for item in history], label="CA")
    plt.plot(epochs, [item["val_cer"] * 100 for item in history], label="CER")
    plt.xlabel("Epoch")
    plt.ylabel("Metric (%)")
    plt.title(label)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path = main_dir / f"fig_main_recognition_{exp_name}_metrics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] saved {output_path}")

    loss_keys = [
        ("val_ctc_loss", "Val CTC"),
        ("val_ortho_loss", "Val Ortho"),
    ]
    if all(key in history[0] for key, _ in loss_keys):
        plt.figure(figsize=(8, 5))
        for key, name in loss_keys:
            plt.plot(epochs, [item[key] for item in history], label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{label} Validation Loss Components")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        output_path = main_dir / f"fig_main_recognition_{exp_name}_loss_components.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[OK] saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/recognition")
    parser.add_argument("--out-dir", type=str, default="outputs/visualizations/paper_figures")
    parser.add_argument("--main-exp", type=str, default="svtr_official_dab_lortho")
    parser.add_argument("--main-label", type=str, default="SVTR + DAB + Lortho")
    parser.add_argument("--no-main-plots", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    plot_metric("train_loss", "Training Loss", "fig_recognition_train_loss.png", root, out_dir, allow_missing=args.allow_missing)
    plot_metric("val_loss", "Validation Loss", "fig_recognition_val_loss.png", root, out_dir, allow_missing=args.allow_missing)
    plot_metric("val_word_accuracy", "Word Accuracy (%)", "fig_recognition_word_accuracy.png", root, out_dir, percent=True, allow_missing=args.allow_missing)
    plot_metric("val_character_accuracy", "Character Accuracy (%)", "fig_recognition_character_accuracy.png", root, out_dir, percent=True, allow_missing=args.allow_missing)
    plot_metric("val_cer", "CER (%)", "fig_recognition_cer.png", root, out_dir, percent=True, allow_missing=args.allow_missing)

    if not args.no_main_plots:
        plot_main_recognition_curves(
            root=root,
            out_dir=out_dir,
            exp_name=args.main_exp,
            label=args.main_label,
            allow_missing=args.allow_missing,
        )


if __name__ == "__main__":
    main()
