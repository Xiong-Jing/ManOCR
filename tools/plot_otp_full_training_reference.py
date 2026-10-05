#!/usr/bin/env python3
"""Plot a full-training OTP-loss reference from measured loss histories.

The source CTC and raw OTP series are read from an actual training history.
The displayed total is deliberately recomputed as

    L_total_reference = L_ctc + lambda_otp * L_otp

It is not silently substituted for the stored training objective when the
source experiment also optimized other auxiliary terms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, LogLocator, MultipleLocator


DEFAULT_METRICS = Path(
    "outputs/metrics/recognition/svtr_dab_otp_lambda_0p1/metrics.json"
)
DEFAULT_OUTPUT_BASE = Path(
    "outputs/visualizations/paper_figures/reference/"
    "fig_svtr_dab_otp_full_training_reference"
)

# Keep the formula total visually primary and identifiable in grayscale.
CURVE_STYLES = {
    "ctc_loss": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.4},
    "otp_loss": {"color": "#009E73", "linestyle": ":", "linewidth": 1.5},
    "formula_total": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.8},
}

DETAIL_Y_MIN = 0.0
DETAIL_Y_MAX = 4.0
FIGURE_TITLE = "SVTR + DAB + OTP Training Loss"
CURVE_LABELS = {
    "ctc_loss": "CTC Loss",
    "otp_loss": "OTP Loss",
    "formula_total": "Total Loss",
}


def _finite_nonnegative(item: dict[str, Any], key: str, index: int) -> float:
    if key not in item:
        raise KeyError(f"History item {index} is missing required field {key!r}.")
    value = float(item[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"History item {index} has invalid {key}={item[key]!r}; "
            "a finite non-negative value is required."
        )
    return value


def load_reference_series(
    metrics_path: Path,
    *,
    require_source_formula: bool = False,
) -> dict[str, Any]:
    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history is empty or invalid: {metrics_path}")

    epochs: list[int] = []
    ctc_losses: list[float] = []
    otp_losses: list[float] = []
    lambdas: list[float] = []
    source_totals: list[float] = []

    for index, item in enumerate(history):
        epoch = int(item.get("epoch", -1))
        if epoch <= 0:
            raise ValueError(f"History item {index} has invalid epoch={epoch}.")
        epochs.append(epoch)
        ctc_losses.append(_finite_nonnegative(item, "train_ctc_loss", index))
        otp_losses.append(_finite_nonnegative(item, "train_ortho_loss", index))
        source_totals.append(_finite_nonnegative(item, "train_loss", index))

        lambda_otp = float(item.get("lambda_ortho", 0.0))
        if not math.isfinite(lambda_otp) or lambda_otp <= 0:
            raise ValueError(
                "OTP must participate from every epoch. "
                f"Found lambda_ortho={lambda_otp!r} at epoch={epoch}."
            )
        lambdas.append(lambda_otp)

    expected_epochs = list(range(epochs[0], epochs[-1] + 1))
    if epochs != expected_epochs:
        raise ValueError(
            "The reference requires one measured record per epoch; "
            f"found epochs {epochs[0]}..{epochs[-1]} with gaps or duplicates."
        )

    lambda_min = min(lambdas)
    lambda_max = max(lambdas)
    if not math.isclose(lambda_min, lambda_max, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "This reference plot expects a constant lambda_ortho, but found "
            f"{lambda_min:.8g}..{lambda_max:.8g}."
        )
    lambda_otp = lambdas[0]

    weighted_otp_losses = [lambda_otp * otp for otp in otp_losses]
    formula_totals = [
        ctc + weighted_otp
        for ctc, weighted_otp in zip(ctc_losses, weighted_otp_losses)
    ]
    source_extra_terms = [
        source - formula
        for source, formula in zip(source_totals, formula_totals)
    ]
    max_formula_error = max(abs(value) for value in source_extra_terms)
    tolerance = max(1e-6, 1e-5 * max(source_totals))
    if require_source_formula and max_formula_error > tolerance:
        raise ValueError(
            "Stored train_loss does not satisfy L_ctc + lambda * L_otp: "
            f"max absolute difference={max_formula_error:.6g}. "
            "The source run contains additional objective terms."
        )

    return {
        "epochs": epochs,
        "ctc_loss": ctc_losses,
        "otp_loss": otp_losses,
        "weighted_otp": weighted_otp_losses,
        "lambda_otp": lambda_otp,
        "formula_total": formula_totals,
        "source_total": source_totals,
        "source_extra_terms": source_extra_terms,
        "max_formula_error": max_formula_error,
        "source_formula_matches": max_formula_error <= tolerance,
    }


def write_source_csv(series: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "ctc_loss_measured",
                "otp_loss_measured",
                "lambda_otp",
                "weighted_otp_contribution",
                "total_loss_formula_reference",
                "source_train_loss",
                "source_additional_terms",
            ]
        )
        for values in zip(
            series["epochs"],
            series["ctc_loss"],
            series["otp_loss"],
            [series["lambda_otp"]] * len(series["epochs"]),
            series["weighted_otp"],
            series["formula_total"],
            series["source_total"],
            series["source_extra_terms"],
        ):
            writer.writerow(values)


def plot_reference(
    series: dict[str, Any],
    output_base: Path,
    *,
    source_label: str,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    full_png_path = output_base.with_name(output_base.name + "_full").with_suffix(".png")
    full_pdf_path = output_base.with_name(output_base.name + "_full").with_suffix(".pdf")
    detail_png_path = output_base.with_name(output_base.name + "_detail").with_suffix(".png")
    detail_pdf_path = output_base.with_name(output_base.name + "_detail").with_suffix(".pdf")
    csv_path = output_base.with_suffix(".csv")
    caption_path = output_base.with_suffix(".caption.txt")

    epochs = series["epochs"]
    lambda_otp = series["lambda_otp"]
    curves = (
        (
            CURVE_LABELS["ctc_loss"],
            series["ctc_loss"],
            CURVE_STYLES["ctc_loss"],
        ),
        (
            CURVE_LABELS["otp_loss"],
            series["otp_loss"],
            CURVE_STYLES["otp_loss"],
        ),
        (
            CURVE_LABELS["formula_total"],
            series["formula_total"],
            CURVE_STYLES["formula_total"],
        ),
    )

    def save_single_figure(
        png_path: Path,
        pdf_path: Path,
        *,
        title: str,
        start_epoch: int,
        log_scale: bool,
    ) -> None:
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for label, values, style in curves:
            selected = [
                (epoch, value)
                for epoch, value in zip(epochs, values)
                if epoch >= start_epoch
            ]
            axis.plot(
                [item[0] for item in selected],
                [item[1] for item in selected],
                label=label,
                **style,
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title(title)
        axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.34)
        axis.legend(frameon=True, fontsize=9)
        axis.set_xlim(max(start_epoch, epochs[0]), epochs[-1])
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        if log_scale:
            axis.set_yscale("log")
            positive_values = [
                value
                for _, values, _ in curves
                for epoch, value in zip(epochs, values)
                if epoch >= start_epoch and value > 0
            ]
            lower = min(positive_values)
            upper = max(positive_values)
            axis.set_ylim(lower / 1.18, upper * 1.18)
            axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            axis.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0))
            )
        else:
            # A zero-based 0--4 range preserves the measured values while making
            # the raw-OTP/total separation visually compact. Fine minor ticks
            # provide detailed reading without fabricating or shifting either curve.
            axis.set_ylim(DETAIL_Y_MIN, DETAIL_Y_MAX)
            axis.yaxis.set_major_locator(MultipleLocator(0.5))
            axis.yaxis.set_minor_locator(MultipleLocator(0.1))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.16)
        figure.tight_layout()
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
        plt.close(figure)

    save_single_figure(
        full_png_path,
        full_pdf_path,
        title=FIGURE_TITLE,
        start_epoch=epochs[0],
        log_scale=True,
    )
    save_single_figure(
        detail_png_path,
        detail_pdf_path,
        title=FIGURE_TITLE,
        start_epoch=max(2, epochs[0]),
        log_scale=False,
    )

    write_source_csv(series, csv_path)
    caption = (
        "Full-training loss reference for SVTR + DAB + OTP. Per-epoch CTC and "
        f"raw OTP losses are measured values from {source_label}. The OTP curve "
        "shows the measured unweighted auxiliary loss, while the displayed "
        f"total is recomputed as L_total = L_CTC + {lambda_otp:g} * L_OTP. "
        "Consequently, raw L_OTP may be numerically above L_total even though only "
        f"{lambda_otp:g} * L_OTP contributes to the total. "
        "It is a formula reference, not the stored total objective of the source "
        "run when that run contains additional loss terms. No smoothing or "
        "synthetic loss values are used."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return (
        full_png_path,
        full_pdf_path,
        detail_png_path,
        detail_pdf_path,
        csv_path,
        caption_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot measured full-training CTC/OTP losses and a total recomputed "
            "strictly as L_ctc + lambda * L_otp."
        )
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument(
        "--source-label",
        default="the svtr_dab_otp_lambda_0p1 experimental run",
    )
    parser.add_argument(
        "--require-source-formula",
        action="store_true",
        help=(
            "Fail unless the stored train_loss already equals the requested "
            "two-term formula. Use this for a future OTP-only rerun."
        ),
    )
    args = parser.parse_args()

    series = load_reference_series(
        args.metrics,
        require_source_formula=args.require_source_formula,
    )
    outputs = plot_reference(
        series,
        args.output_base,
        source_label=args.source_label,
    )

    if not series["source_formula_matches"]:
        print(
            "[NOTICE] The source run contains additional objective terms; "
            "the displayed total was recomputed by the requested two-term "
            "formula. max_difference="
            f"{series['max_formula_error']:.6g}"
        )
    for output in outputs:
        print(f"[OK] saved {output}")


if __name__ == "__main__":
    main()
