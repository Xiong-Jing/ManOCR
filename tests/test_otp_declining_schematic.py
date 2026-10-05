import csv
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "tools" / "plot_otp_declining_schematic.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("otp_declining_schematic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _history():
    return [
        {
            "epoch": epoch,
            "train_ctc_loss": 1.8 / (epoch ** 0.4),
            "train_ortho_loss": 2.4 - 0.002 * epoch,
            "lambda_ortho": 0.1,
        }
        for epoch in range(1, 201)
    ]


def test_tail_is_strictly_declining_and_formula_consistent(tmp_path):
    module = _load_module()
    assert module.FIGURE_TITLE == "SVTR + DAB + OTP Training Loss"
    assert module.CURVE_LABELS == {
        "ctc_loss": "CTC Loss",
        "otp_loss": "OTP Loss",
        "formula_total": "Total Loss",
    }
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_history()), encoding="utf-8")
    measured = module.load_measured_history(metrics)
    series = module.build_declining_schematic(measured)

    start = series["epochs"].index(100)
    tail = series["otp_loss"][start:]
    assert all(left > right for left, right in zip(tail, tail[1:]))
    assert series["otp_source"][start] == "measured"
    assert set(series["otp_source"][start + 1 :]) == {"schematic"}
    assert series["otp_loss"][-1] == pytest.approx(
        series["otp_loss"][start] * 0.60
    )
    assert series["formula_total"] == pytest.approx(
        [
            ctc + 0.1 * otp
            for ctc, otp in zip(series["ctc_loss"], series["otp_loss"])
        ]
    )


def test_plot_exports_explicit_schematic_disclosure(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_module()
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_history()), encoding="utf-8")
    series = module.build_declining_schematic(module.load_measured_history(metrics))
    outputs = module.plot_schematic(series, tmp_path / "schematic")

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    caption = outputs[3].read_text(encoding="utf-8")
    assert "示意图，非实验结果" in caption
    assert "不能作为实验结果" in caption
    with outputs[2].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[99]["otp_source"] == "measured"
    assert rows[100]["otp_source"] == "schematic"


def test_invalid_tail_parameters_are_rejected(tmp_path):
    module = _load_module()
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_history()), encoding="utf-8")
    measured = module.load_measured_history(metrics)
    with pytest.raises(ValueError, match="between 0 and 1"):
        module.build_declining_schematic(measured, tail_end_ratio=1.0)
