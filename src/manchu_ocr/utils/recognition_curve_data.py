from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def extract_metric_series(
    history: Sequence[Mapping[str, object]],
    metric_key: str,
    *,
    percent: bool = False,
) -> dict[str, list[float]]:
    """Keep only measured, finite values at their actual epoch numbers.

    Skipped validation epochs contain NaN values. They must not become zeros,
    repeated measurements, or fabricated per-epoch validation scores.
    """
    epochs: list[float] = []
    values: list[float] = []
    for item in history:
        value = item.get(metric_key)
        if value is None:
            continue
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            continue
        epochs.append(float(item["epoch"]))
        values.append(numeric_value * 100 if percent else numeric_value)
    return {"epochs": epochs, "values": values}


def extract_training_loss_series(
    history: Sequence[Mapping[str, object]],
) -> dict[str, list[float]]:
    """Return the three training-loss curves used by recognition figures.

    ``otp_loss`` is the raw orthographic transition-penalty objective. The
    optimized total still uses ``lambda_ortho * otp_loss`` and may also contain
    the configured alignment or NRTR auxiliary terms. The weighted OTP
    contribution is retained separately in training history for auditing but is
    intentionally not substituted for the requested OTP-loss curve.

    Histories written before ``train_ortho_loss`` was introduced can only be
    plotted safely when OTP was inactive. Missing OTP values from an epoch with
    a positive lambda raise an error instead of being silently shown as zero.
    """
    if not history:
        raise ValueError("Recognition training history is empty.")

    epochs: list[float] = []
    ctc_loss: list[float] = []
    otp_loss: list[float] = []
    total_loss: list[float] = []

    for index, item in enumerate(history):
        for key in ("epoch", "train_ctc_loss", "train_loss"):
            if key not in item:
                raise KeyError(
                    f"Recognition history item {index} is missing required field {key!r}."
                )

        lambda_ortho = float(item.get("lambda_ortho", 0.0))
        if "train_ortho_loss" in item:
            current_otp_loss = float(item["train_ortho_loss"])
        elif abs(lambda_ortho) <= 1e-15:
            current_otp_loss = 0.0
        else:
            epoch = item["epoch"]
            raise KeyError(
                "Recognition history predates training OTP-loss logging: "
                f"epoch={epoch}, lambda_ortho={lambda_ortho}, missing "
                "'train_ortho_loss'. Rerun training to produce a truthful full curve."
            )

        epochs.append(float(item["epoch"]))
        ctc_loss.append(float(item["train_ctc_loss"]))
        otp_loss.append(current_otp_loss)
        total_loss.append(float(item["train_loss"]))

    return {
        "epochs": epochs,
        "ctc_loss": ctc_loss,
        "otp_loss": otp_loss,
        "total_loss": total_loss,
    }
