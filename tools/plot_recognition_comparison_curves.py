import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from manchu_ocr.utils.recognition_curve_data import (
    extract_metric_series,
    extract_training_loss_series,
)


EXPERIMENTS = [
    ("CRNN", "crnn_baseline"),
    ("PARSeq", "parseq_baseline"),
    ("ABINet", "abinet_baseline"),
    ("SVTRv2", "svtrv2_baseline"),
    ("SVTRv2 + NRTR", "svtrv2_nrtr_baseline"),
    ("DCM", "dcm_baseline"),
    ("Ours", "svtr_official_dab_lortho"),
]


def load_history(root: Path, exp_name: str) -> list[dict]:
    path = root / exp_name / "metrics.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric(
    root: Path,
    out_dir: Path,
    metric_key: str,
    ylabel: str,
    output_name: str,   
    percent: bool = False,
    allow_missing: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plotted = False

    for label, exp_name in EXPERIMENTS:
        path = root / exp_name / "metrics.json"

        if allow_missing and not path.exists():
            continue

        history = load_history(root, exp_name)
        series = extract_metric_series(history, metric_key, percent=percent)
        if not series["values"]:
            continue
        plt.plot(series["epochs"], series["values"], label=label, marker="o")
        plotted = True

    if not plotted:
        plt.close()
        return

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
    """Plot CTC, raw OTP, and optimized total loss in per-model panels."""
    available: list[tuple[str, list[dict]]] = []
    for label, exp_name in EXPERIMENTS:
        history_path = root / exp_name / "metrics.json"
        if allow_missing and not history_path.exists():
            continue
        available.append((label, load_history(root, exp_name)))

    if not available:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = 2
    nrows = (len(available) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 4.2 * nrows),
        squeeze=False,
    )
    flat_axes = [axis for row in axes for axis in row]

    for axis, (label, history) in zip(flat_axes, available):
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

    figure.suptitle("Recognition Comparison Training Loss Components")
    figure.tight_layout()
    output_path = out_dir / "fig_recognition_comparison_train_loss.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[OK] saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/recognition")
    parser.add_argument("--out-dir", type=str, default="outputs/visualizations/paper_figures")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    plot_training_loss_components(
        root=root,
        out_dir=out_dir,
        allow_missing=args.allow_missing,
    )
    plot_metric(
        root=root,
        out_dir=out_dir,
        metric_key="val_loss",
        ylabel="Validation Loss",
        output_name="fig_recognition_comparison_val_loss.png",
        allow_missing=args.allow_missing,
    )
    plot_metric(
        root=root,
        out_dir=out_dir,
        metric_key="val_word_accuracy",
        ylabel="Validation WA (%)",
        output_name="fig_recognition_comparison_val_wa.png",
        percent=True,
        allow_missing=args.allow_missing,
    )
    plot_metric(
        root=root,
        out_dir=out_dir,
        metric_key="val_character_accuracy",
        ylabel="Validation CA (%)",
        output_name="fig_recognition_comparison_val_ca.png",
        percent=True,
        allow_missing=args.allow_missing,
    )
    plot_metric(
        root=root,
        out_dir=out_dir,
        metric_key="val_cer",
        ylabel="Validation CER (%)",
        output_name="fig_recognition_comparison_val_cer.png",
        percent=True,
        allow_missing=args.allow_missing,
    )


if __name__ == "__main__":
    main()
