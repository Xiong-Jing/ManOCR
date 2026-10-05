import csv
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "tools" / "plot_otp_full_training_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("otp_loss_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _history():
    return [
        {
            "epoch": epoch,
            "train_ctc_loss": 2.0 / epoch,
            "train_ortho_loss": 1.0 / (epoch ** 0.5),
            "lambda_ortho": 0.1,
            "train_loss": 2.0 / epoch + 0.1 / (epoch ** 0.5) + 0.02,
        }
        for epoch in range(1, 6)
    ]


def test_reference_uses_measured_components_and_requested_formula(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_module()
    assert module.CURVE_STYLES["formula_total"] == {
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.8,
    }
    assert module.CURVE_STYLES["otp_loss"] == {
        "color": "#009E73",
        "linestyle": ":",
        "linewidth": 1.5,
    }
    assert module.DETAIL_Y_MIN == 0.0
    assert module.DETAIL_Y_MAX == 4.0
    assert module.FIGURE_TITLE == "SVTR + DAB + OTP Training Loss"
    assert module.CURVE_LABELS == {
        "ctc_loss": "CTC Loss",
        "otp_loss": "OTP Loss",
        "formula_total": "Total Loss",
    }
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_history()), encoding="utf-8")

    series = module.load_reference_series(metrics)
    assert series["lambda_otp"] == 0.1
    assert series["ctc_loss"] == [2.0 / epoch for epoch in range(1, 6)]
    assert series["otp_loss"] == [1.0 / (epoch ** 0.5) for epoch in range(1, 6)]
    assert series["weighted_otp"] == pytest.approx(
        [0.1 / (epoch ** 0.5) for epoch in range(1, 6)]
    )
    assert series["formula_total"] == pytest.approx(
        [
            2.0 / epoch + 0.1 / (epoch ** 0.5)
            for epoch in range(1, 6)
        ]
    )
    assert series["source_formula_matches"] is False

    outputs = module.plot_reference(
        series,
        tmp_path / "figure",
        source_label="a test experiment",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    assert {path.suffix for path in outputs[:4]} == {".png", ".pdf"}
    assert outputs[0].name.endswith("_full.png")
    assert outputs[2].name.endswith("_detail.png")
    with outputs[4].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert float(rows[-1]["total_loss_formula_reference"]) == pytest.approx(
        series["formula_total"][-1]
    )
    assert float(rows[-1]["weighted_otp_contribution"]) == pytest.approx(
        series["weighted_otp"][-1]
    )
    caption = outputs[5].read_text(encoding="utf-8")
    assert "measured unweighted auxiliary loss" in caption
    assert "not the stored total objective" in caption


def test_rejects_delayed_otp_activation(tmp_path):
    module = _load_module()
    history = _history()
    history[0]["lambda_ortho"] = 0.0
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(history), encoding="utf-8")
    with pytest.raises(ValueError, match="participate from every epoch"):
        module.load_reference_series(metrics)


def test_strict_mode_rejects_unreported_additional_terms(tmp_path):
    module = _load_module()
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_history()), encoding="utf-8")
    with pytest.raises(ValueError, match="additional objective terms"):
        module.load_reference_series(metrics, require_source_formula=True)
