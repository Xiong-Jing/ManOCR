#!/usr/bin/env python3
"""Create an explicitly labelled, non-experimental OTP-loss schematic.

The measured CTC/OTP history is retained through ``tail_start_epoch``.  After
that boundary, the OTP curve is replaced by a strictly decreasing illustrative
tail and the displayed total is recomputed as

    L_total = L_ctc + lambda_otp * L_otp_schematic.

This output is a visual reference only.  It must not be reported as an
experimental training curve.
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
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator


DEFAULT_METRICS = Path(
    "outputs/metrics/recognition/svtr_dab_otp_lambda_0p1/metrics.json"
)
DEFAULT_OUTPUT_BASE = Path(
    "outputs/visualizations/paper_figures/reference/"
    "fig_svtr_dab_otp_declining_schematic"
)

CURVE_STYLES = {
    "ctc_loss": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.4},
    "otp_loss": {"color": "#009E73", "linestyle": ":", "linewidth": 1.5},
    "formula_total": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.8},
}
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


def load_measured_history(metrics_path: Path) -> dict[str, Any]:
    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history is empty or invalid: {metrics_path}")

    epochs: list[int] = []
    ctc_losses: list[float] = []
    otp_losses: list[float] = []
    lambdas: list[float] = []
    for index, item in enumerate(history):
        epoch = int(item.get("epoch", -1))
        if epoch <= 0:
            raise ValueError(f"History item {index} has invalid epoch={epoch}.")
        epochs.append(epoch)
        ctc_losses.append(_finite_nonnegative(item, "train_ctc_loss", index))
        otp_losses.append(_finite_nonnegative(item, "train_ortho_loss", index))
        lambda_otp = float(item.get("lambda_ortho", 0.0))
        if not math.isfinite(lambda_otp) or not 0 < lambda_otp < 1:
            raise ValueError(
                f"Expected 0 < lambda_ortho < 1, found {lambda_otp!r} "
                f"at epoch {epoch}."
            )
        lambdas.append(lambda_otp)

    if epochs != list(range(epochs[0], epochs[-1] + 1)):
        raise ValueError("The schematic requires one measured record per epoch.")
    if not math.isclose(min(lambdas), max(lambdas), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The schematic requires a constant lambda_ortho.")

    return {
        "epochs": epochs,
        "ctc_loss": ctc_losses,
        "otp_loss": otp_losses,
        "lambda_otp": lambdas[0],
    }


def build_declining_schematic(
    measured: dict[str, Any],
    *,
    tail_start_epoch: int = 100,
    tail_end_ratio: float = 0.60,
    curve_power: float = 1.15,
) -> dict[str, Any]:
    """Replace the late OTP segment with a strictly decreasing reference tail."""
    epochs = list(measured["epochs"])
    ctc_losses = list(measured["ctc_loss"])
    measured_otp = list(measured["otp_loss"])
    lambda_otp = float(measured["lambda_otp"])

    if tail_start_epoch not in epochs:
        raise ValueError(f"tail_start_epoch={tail_start_epoch} is not in the history.")
    if tail_start_epoch >= epochs[-1]:
        raise ValueError("tail_start_epoch must be earlier than the last epoch.")
    if not 0 < tail_end_ratio < 1:
        raise ValueError("tail_end_ratio must be between 0 and 1.")
    if curve_power <= 0:
        raise ValueError("curve_power must be positive.")

    start_index = epochs.index(tail_start_epoch)
    start_value = measured_otp[start_index]
    end_value = start_value * tail_end_ratio
    if end_value <= 0:
        raise ValueError("The schematic tail endpoint must be positive.")

    schematic_otp = measured_otp[: start_index + 1]
    tail_span = epochs[-1] - tail_start_epoch
    for epoch in epochs[start_index + 1 :]:
        progress = (epoch - tail_start_epoch) / tail_span
        value = end_value + (start_value - end_value) * (1.0 - progress) ** curve_power
        schematic_otp.append(value)

    tail_values = schematic_otp[start_index:]
    if not all(left > right for left, right in zip(tail_values, tail_values[1:])):
        raise AssertionError("Constructed OTP tail is not strictly decreasing.")

    formula_total = [
        ctc + lambda_otp * otp
        for ctc, otp in zip(ctc_losses, schematic_otp)
    ]
    visible_start_index = next(
        (index for index, epoch in enumerate(epochs) if epoch >= 2), 0
    )
    if not all(
        otp > total
        for otp, total in zip(
            schematic_otp[visible_start_index:],
            formula_total[visible_start_index:],
        )
    ):
        raise ValueError(
            "The requested schematic cannot keep raw OTP above formula total "
            "with the selected tail parameters."
        )

    sources = [
        "measured" if epoch <= tail_start_epoch else "schematic"
        for epoch in epochs
    ]
    return {
        "epochs": epochs,
        "ctc_loss": ctc_losses,
        "otp_loss": schematic_otp,
        "measured_otp_loss": measured_otp,
        "formula_total": formula_total,
        "lambda_otp": lambda_otp,
        "tail_start_epoch": tail_start_epoch,
        "tail_end_ratio": tail_end_ratio,
        "curve_power": curve_power,
        "otp_source": sources,
    }


def _chinese_font() -> FontProperties | None:
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if family in available:
            return FontProperties(family=family)
    return None


def write_source_csv(series: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "ctc_loss_measured",
                "otp_loss_measured",
                "otp_loss_displayed",
                "otp_source",
                "lambda_otp",
                "total_loss_formula_schematic",
            ]
        )
        for row in zip(
            series["epochs"],
            series["ctc_loss"],
            series["measured_otp_loss"],
            series["otp_loss"],
            series["otp_source"],
            [series["lambda_otp"]] * len(series["epochs"]),
            series["formula_total"],
        ):
            writer.writerow(row)


def plot_schematic(
    series: dict[str, Any], output_base: Path
) -> tuple[Path, Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    csv_path = output_base.with_suffix(".csv")
    caption_path = output_base.with_suffix(".caption.txt")

    start_index = next(
        (index for index, epoch in enumerate(series["epochs"]) if epoch >= 2), 0
    )
    epochs = series["epochs"][start_index:]
    ctc = series["ctc_loss"][start_index:]
    otp = series["otp_loss"][start_index:]
    total = series["formula_total"][start_index:]
    lambda_otp = series["lambda_otp"]
    boundary = series["tail_start_epoch"]

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.axvspan(
        boundary,
        epochs[-1],
        color="#CC79A7",
        alpha=0.07,
        zorder=0,
    )
    axis.axvline(boundary, color="#7A7A7A", linewidth=0.9, linestyle="-.")
    axis.plot(
        epochs,
        ctc,
        label=CURVE_LABELS["ctc_loss"],
        **CURVE_STYLES["ctc_loss"],
    )
    axis.plot(
        epochs,
        otp,
        label=CURVE_LABELS["otp_loss"],
        **CURVE_STYLES["otp_loss"],
    )
    axis.plot(
        epochs,
        total,
        label=CURVE_LABELS["formula_total"],
        **CURVE_STYLES["formula_total"],
    )

    axis.set_xlim(epochs[0], epochs[-1])
    axis.set_ylim(0.0, 4.0)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(FIGURE_TITLE)
    axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    axis.yaxis.set_major_locator(MultipleLocator(0.5))
    axis.yaxis.set_minor_locator(MultipleLocator(0.1))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.34)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.16)
    axis.legend(frameon=True, fontsize=8.2, loc="upper right")

    warning_cn = "示意图，非实验结果"
    warning_en = "SCHEMATIC — NOT AN EXPERIMENTAL RESULT"
    chinese_font = _chinese_font()
    warning = f"{warning_cn} / {warning_en}" if chinese_font else warning_en
    figure.text(
        0.5,
        0.968,
        warning,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#B22222",
        fontweight="bold",
        fontproperties=chinese_font,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "#FFF2F2", "alpha": 0.95},
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    write_source_csv(series, csv_path)
    caption = (
        "示意图，非实验结果。Epoch 2–100 的 CTC 和 OTP 来自测量训练历史；"
        f"Epoch {boundary + 1}–{series['epochs'][-1]} 的 OTP 是严格递减的构造参考尾段，"
        f"终点为边界值的 {series['tail_end_ratio']:.0%}。CTC 保持测量值，"
        f"total 按 L_total = L_CTC + {lambda_otp:g} * L_OTP_schematic 重新计算。"
        "该图只能用于趋势设计或方法说明，不能作为实验结果、收敛证据或论文数值报告。"
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return png_path, pdf_path, csv_path, caption_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clearly labelled non-experimental declining OTP schematic."
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--tail-start-epoch", type=int, default=100)
    parser.add_argument("--tail-end-ratio", type=float, default=0.60)
    parser.add_argument("--curve-power", type=float, default=1.15)
    args = parser.parse_args()

    measured = load_measured_history(args.metrics)
    series = build_declining_schematic(
        measured,
        tail_start_epoch=args.tail_start_epoch,
        tail_end_ratio=args.tail_end_ratio,
        curve_power=args.curve_power,
    )
    outputs = plot_schematic(series, args.output_base)
    for output in outputs:
        print(f"[OK] saved {output}")
    print("[WARNING] 示意图，非实验结果；不得作为实验测量曲线报告。")


if __name__ == "__main__":
    main()
