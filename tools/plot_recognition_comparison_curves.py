import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENTS = [
    ("CRNN", "crnn_baseline"),
    ("PARSeq", "parseq_baseline"),
    ("ABINet", "abinet_baseline"),
    ("SVTRv2", "svtrv2_baseline"),
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
        values = []
        epochs = []

        for item in history:
            if metric_key not in item:
                continue

            value = item[metric_key]

            if value != value:
                continue

            epochs.append(item["epoch"])
            values.append(value * 100 if percent else value)

        if not values:
            continue

        plt.plot(epochs, values, label=label)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/recognition")
    parser.add_argument("--out-dir", type=str, default="outputs/visualizations/paper_figures")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    plot_metric(
        root=root,
        out_dir=out_dir,
        metric_key="train_loss",
        ylabel="Training Loss",
        output_name="fig_recognition_comparison_train_loss.png",
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
