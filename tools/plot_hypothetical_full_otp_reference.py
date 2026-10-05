"""Plot an explicitly hypothetical full-process OTP-loss reference.

The measured CTC and total losses are copied unchanged from ``metrics.json``.
The OTP curve is a deterministic, synthetic reference trajectory; it is not a
measurement and must not be reported as an experimental result.  A sidecar CSV
and JSON provenance record are written with every figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS = Path(
    "outputs/metrics/recognition/svtr_official_dab_lortho/metrics.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/visualizations/reference_only")
DEFAULT_STEM = "fig_hypothetical_full_process_otp_reference_NOT_EXPERIMENTAL"


def _load_measured_losses(path: Path) -> tuple[list[float], list[float], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) < 2:
        raise ValueError("metrics.json must contain at least two epoch records.")

    epochs: list[float] = []
    ctc_loss: list[float] = []
    total_loss: list[float] = []
    for index, item in enumerate(history):
        try:
            epoch = float(item["epoch"])
            ctc = float(item["train_ctc_loss"])
            total = float(item["train_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid training-loss record at index {index}.") from exc
        if not all(math.isfinite(value) and value > 0 for value in (epoch, ctc, total)):
            raise ValueError(
                f"Epoch, CTC loss, and total loss must be finite and positive: index={index}."
            )
        epochs.append(epoch)
        ctc_loss.append(ctc)
        total_loss.append(total)

    if any(right <= left for left, right in zip(epochs, epochs[1:])):
        raise ValueError("Epochs must be strictly increasing.")
    return epochs, ctc_loss, total_loss


def synthetic_otp_reference(
    epochs: Sequence[float],
    *,
    start: float,
    end: float,
    tau: float,
) -> list[float]:
    """Return a normalized exponential reference that spans all epochs.

    ``start``, ``end`` and ``tau`` are illustration parameters, not fitted
    experimental estimates.  Normalization makes the first and final values
    exactly equal to ``start`` and ``end``.
    """
    if len(epochs) < 2:
        raise ValueError("At least two epochs are required.")
    if not (math.isfinite(start) and math.isfinite(end) and math.isfinite(tau)):
        raise ValueError("OTP reference parameters must be finite.")
    if not (start > end > 0 and tau > 0):
        raise ValueError("Expected start > end > 0 and tau > 0.")

    first = float(epochs[0])
    duration = float(epochs[-1]) - first
    if duration <= 0:
        raise ValueError("Epoch range must be positive.")
    terminal = math.exp(-duration / tau)
    scale = 1.0 - terminal
    return [
        end
        + (start - end)
        * (math.exp(-(float(epoch) - first) / tau) - terminal)
        / scale
        for epoch in epochs
    ]


def generate_reference_figure(
    metrics_path: Path,
    output_dir: Path,
    *,
    otp_start: float = 3.5,
    otp_end: float = 0.35,
    otp_tau: float = 52.0,
    output_stem: str = DEFAULT_STEM,
) -> dict[str, Path]:
    """Generate the labeled reference figure and auditable sidecar files."""
    epochs, ctc_loss, total_loss = _load_measured_losses(metrics_path)
    otp_reference = synthetic_otp_reference(
        epochs,
        start=otp_start,
        end=otp_end,
        tau=otp_tau,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8.4, 5.3))
    axis.plot(
        epochs,
        ctc_loss,
        color="#0072B2",
        linewidth=1.8,
        label="Measured CTC loss",
    )
    axis.plot(
        epochs,
        otp_reference,
        color="#D55E00",
        linewidth=1.8,
        linestyle="--",
        label="Synthetic OTP reference (not measured)",
    )
    axis.plot(
        epochs,
        total_loss,
        color="#009E73",
        linewidth=2.0,
        linestyle="-.",
        label="Measured total loss",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss (log scale)")
    axis.set_title("Hypothetical Full-Process OTP Optimization Reference")
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.45)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.25)
    axis.legend(loc="upper right", frameon=True)
    axis.text(
        0.015,
        0.025,
        "REFERENCE ONLY — OTP trajectory is synthetic, not experimental evidence",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#9C2F2F",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9,
              "edgecolor": "#9C2F2F"},
    )
    figure.tight_layout()

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    csv_path = output_dir / f"{output_stem}.csv"
    json_path = output_dir / f"{output_stem}.json"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch", "measured_ctc_loss", "synthetic_otp_reference", "measured_total_loss"]
        )
        writer.writerows(zip(epochs, ctc_loss, otp_reference, total_loss))

    provenance = {
        "figure_type": "hypothetical_reference",
        "not_experimental_result": True,
        "measured_source": str(metrics_path.resolve()),
        "measured_series": {
            "ctc_loss": "train_ctc_loss",
            "total_loss": "train_loss",
        },
        "synthetic_series": {
            "name": "synthetic_otp_reference",
            "formula": (
                "end + (start-end) * (exp(-(epoch-first)/tau) - "
                "exp(-duration/tau)) / (1-exp(-duration/tau))"
            ),
            "parameters": {
                "start": otp_start,
                "end": otp_end,
                "tau_epochs": otp_tau,
            },
            "interpretation": (
                "Illustrative full-process decreasing OTP trajectory only; "
                "not observed, fitted, or valid for quantitative claims."
            ),
        },
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    json_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"png": png_path, "pdf": pdf_path, "csv": csv_path, "json": json_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a clearly labeled hypothetical OTP reference beside measured "
            "CTC/total curves. This does not create experimental OTP evidence."
        )
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--otp-start", type=float, default=3.5)
    parser.add_argument("--otp-end", type=float, default=0.35)
    parser.add_argument("--otp-tau", type=float, default=52.0)
    parser.add_argument("--output-stem", default=DEFAULT_STEM)
    args = parser.parse_args()

    outputs = generate_reference_figure(
        metrics_path=args.metrics,
        output_dir=args.output_dir,
        otp_start=args.otp_start,
        otp_end=args.otp_end,
        otp_tau=args.otp_tau,
        output_stem=args.output_stem,
    )
    for kind, path in outputs.items():
        print(f"[OK] {kind}: {path}")


if __name__ == "__main__":
    main()
