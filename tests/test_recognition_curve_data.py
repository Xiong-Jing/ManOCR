import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

from manchu_ocr.utils.recognition_curve_data import (
    extract_metric_series,
    extract_training_loss_series,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_extracts_ctc_otp_and_total_training_losses():
    history = [
        {
            "epoch": 1,
            "lambda_ortho": 0.1,
            "train_ctc_loss": 2.0,
            "train_ortho_loss": 0.4,
            "train_loss": 2.1,
        },
        {
            "epoch": 2,
            "lambda_ortho": 0.1,
            "train_ctc_loss": 1.5,
            "train_ortho_loss": 0.3,
            "train_loss": 1.58,
        },
    ]

    assert extract_training_loss_series(history) == {
        "epochs": [1.0, 2.0],
        "ctc_loss": [2.0, 1.5],
        "otp_loss": [0.4, 0.3],
        "total_loss": [2.1, 1.58],
    }


def test_old_non_otp_history_uses_zero_otp_curve():
    history = [
        {
            "epoch": 1,
            "lambda_ortho": 0.0,
            "train_ctc_loss": 2.0,
            "train_loss": 2.02,
        }
    ]
    assert extract_training_loss_series(history)["otp_loss"] == [0.0]


def test_old_positive_otp_history_is_not_silently_misplotted():
    history = [
        {
            "epoch": 40,
            "lambda_ortho": 0.1,
            "train_ctc_loss": 1.0,
            "train_loss": 1.2,
        }
    ]

    try:
        extract_training_loss_series(history)
    except KeyError as exc:
        assert "predates training OTP-loss logging" in str(exc)
    else:
        raise AssertionError("Missing positive-lambda OTP history must raise KeyError.")


def test_sparse_validation_series_keeps_only_real_measurements():
    history = [
        {"epoch": 1, "val_cer": float("nan")},
        {"epoch": 49, "val_cer": None},
        {"epoch": 50, "val_cer": 0.2},
        {"epoch": 51},
        {"epoch": 99, "val_cer": float("inf")},
        {"epoch": 100, "val_cer": 0.1},
        {"epoch": 150, "val_cer": 0.08},
        {"epoch": 200, "val_cer": 0.05},
    ]
    assert extract_metric_series(history, "val_cer", percent=True) == {
        "epochs": [50.0, 100.0, 150.0, 200.0],
        "values": [20.0, 10.0, 8.0, 5.0],
    }
    assert extract_metric_series(history, "missing") == {"epochs": [], "values": []}


def test_old_dense_history_remains_dense():
    history = [{"epoch": epoch, "val_loss": 1.0 / epoch} for epoch in range(1, 4)]
    assert extract_metric_series(history, "val_loss") == {
        "epochs": [1.0, 2.0, 3.0],
        "values": [1.0, 0.5, 1.0 / 3],
    }


@pytest.mark.parametrize("script", [
    "plot_recognition_comparison_curves.py",
    "plot_recognition_formal_curves.py",
    "plot_final_recognition_model.py",
])
def test_plotters_render_sparse_validation_and_preserve_training_losses(
    script, tmp_path, monkeypatch
):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = tmp_path / "metrics"
    experiment = root / "sparse"
    experiment.mkdir(parents=True)
    history = []
    for epoch in range(1, 201):
        validated = epoch % 50 == 0
        value = 0.5 if validated else float("nan")
        history.append({
            "epoch": epoch,
            "lambda_ortho": 0.1,
            "train_ctc_loss": 2.0 / epoch,
            "train_ortho_loss": 0.1 / epoch,
            "train_loss": 2.01 / epoch,
            "val_word_accuracy": value,
            "val_character_accuracy": value,
            "val_cer": value,
            "val_ctc_loss": value,
            "val_ortho_loss": value,
            "val_align_loss": value,
        })
    (experiment / "metrics.json").write_text(json.dumps(history), encoding="utf-8")
    namespace = runpy.run_path(str(PROJECT_ROOT / "tools" / script))
    plot = Mock(wraps=plt.plot)
    monkeypatch.setattr(plt, "plot", plot)
    out_dir = tmp_path / "figures"

    if script == "plot_recognition_comparison_curves.py":
        function = namespace["plot_metric"]
        function.__globals__["EXPERIMENTS"] = [("Sparse", "sparse")]
        function(root, out_dir, "val_word_accuracy", "WA", "sparse.png", percent=True)
        validation_label = "Sparse"
    elif script == "plot_recognition_formal_curves.py":
        namespace["plot_main_recognition_curves"](root, out_dir, exp_name="sparse")
        validation_label = "WA"
    else:
        namespace["plot_train_val_curves"](root, out_dir, "sparse", "Sparse")
        validation_label = "Val WA"

    validation_call = next(
        call for call in plot.call_args_list
        if call.kwargs.get("label") == validation_label
    )
    assert validation_call.args[0] == [50.0, 100.0, 150.0, 200.0]
    assert validation_call.args[1] == [50.0] * 4
    assert validation_call.kwargs["marker"] == "o"
    if script != "plot_recognition_comparison_curves.py":
        for label in ("CTC Loss", "OTP Loss", "Total Loss"):
            call = next(
                call for call in plot.call_args_list
                if call.kwargs.get("label") == label
            )
            assert call.args[0] == list(range(1, 201))
    assert list(out_dir.rglob("*.png"))
