import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from manchu_ocr.utils.recognition_curve_data import (
    extract_metric_series,
    extract_training_loss_series,
)


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
        series = extract_metric_series(history, metric_key, percent=percent)
        if not series["values"]:
            continue
        plt.plot(series["epochs"], series["values"], label=label, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = out_dir / output_name
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[OK] saved {output_path}")


def plot_training_loss_components(
    root: Path,
    out_dir: Path,
    allow_missing: bool = False,
) -> None:
    """Plot CTC, raw OTP, and optimized total loss for every ablation model."""
    available: list[tuple[str, str, list[dict]]] = []
    for label, exp_name in EXPERIMENTS:
        history_path = root / exp_name / "metrics.json"
        if allow_missing and not history_path.exists():
            continue
        available.append((label, exp_name, load_history(exp_name, root=root)))

    if not available:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = 1 if len(available) == 1 else 2
    nrows = (len(available) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 4.5 * nrows),
        squeeze=False,
    )
    flat_axes = [axis for row in axes for axis in row]

    for axis, (label, _exp_name, history) in zip(flat_axes, available):
        series = extract_training_loss_series(history)
        axis.plot(series["epochs"], series["ctc_loss"], label="CTC Loss")
        axis.plot(series["epochs"], series["otp_loss"], label="OTP Loss")
        axis.plot(
            series["epochs"],
            series["total_loss"],
            label="Total Loss",
            linewidth=2.0,
        )
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Training Loss")
        axis.legend()
        axis.grid(True, linestyle="--", alpha=0.4)

    for axis in flat_axes[len(available):]:
        axis.set_visible(False)

    figure.suptitle("Recognition Training Loss Components")
    figure.tight_layout()
    output_path = out_dir / "fig_recognition_train_loss.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
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

    main_dir = out_dir / "main_models"
    main_dir.mkdir(parents=True, exist_ok=True)

    training_losses = extract_training_loss_series(history)
    plt.figure(figsize=(8, 5))
    plt.plot(
        training_losses["epochs"],
        training_losses["ctc_loss"],
        label="CTC Loss",
    )
    plt.plot(
        training_losses["epochs"],
        training_losses["otp_loss"],
        label="OTP Loss",
    )
    plt.plot(
        training_losses["epochs"],
        training_losses["total_loss"],
        label="Total Loss",
        linewidth=2.0,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{label} Training Loss Components")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path = main_dir / f"fig_main_recognition_{exp_name}_loss.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] saved {output_path}")

    plt.figure(figsize=(8, 5))
    for key, name in (
        ("val_word_accuracy", "WA"),
        ("val_character_accuracy", "CA"),
        ("val_cer", "CER"),
    ):
        series = extract_metric_series(history, key, percent=True)
        if series["values"]:
            plt.plot(series["epochs"], series["values"], label=name, marker="o")
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
        ("val_ortho_loss", "Val OTP"),
    ]
    if all(key in history[0] for key, _ in loss_keys):
        plt.figure(figsize=(8, 5))
        for key, name in loss_keys:
            series = extract_metric_series(history, key)
            if series["values"]:
                plt.plot(series["epochs"], series["values"], label=name, marker="o")
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

    plot_training_loss_components(
        root=root,
        out_dir=out_dir,
        allow_missing=args.allow_missing,
    )
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
