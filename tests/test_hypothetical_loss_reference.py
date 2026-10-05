import csv
import json
import runpy
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "tools" / "plot_hypothetical_full_otp_reference.py"


def _history():
    return [
        {"epoch": 1, "train_ctc_loss": 4.0, "train_loss": 4.5},
        {"epoch": 2, "train_ctc_loss": 2.0, "train_loss": 2.4},
        {"epoch": 3, "train_ctc_loss": 1.0, "train_loss": 1.3},
        {"epoch": 4, "train_ctc_loss": 0.5, "train_loss": 0.8},
    ]


def test_reference_curve_spans_full_training_and_is_explicitly_synthetic(tmp_path):
    pytest.importorskip("matplotlib")
    namespace = runpy.run_path(str(SCRIPT))
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_history()), encoding="utf-8")

    outputs = namespace["generate_reference_figure"](
        metrics,
        tmp_path / "reference",
        otp_start=3.0,
        otp_end=0.3,
        otp_tau=2.0,
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())

    with outputs["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["measured_ctc_loss"]) for row in rows] == [4.0, 2.0, 1.0, 0.5]
    assert [float(row["measured_total_loss"]) for row in rows] == [4.5, 2.4, 1.3, 0.8]
    otp = [float(row["synthetic_otp_reference"]) for row in rows]
    assert otp[0] == pytest.approx(3.0)
    assert otp[-1] == pytest.approx(0.3)
    assert all(left > right > 0 for left, right in zip(otp, otp[1:]))
    assert otp != [4.0, 2.0, 1.0, 0.5]
    assert otp != [4.5, 2.4, 1.3, 0.8]

    provenance = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert provenance["figure_type"] == "hypothetical_reference"
    assert provenance["not_experimental_result"] is True
    assert "not observed" in provenance["synthetic_series"]["interpretation"]


@pytest.mark.parametrize(
    "start,end,tau",
    [(0.3, 0.3, 2.0), (0.2, 0.3, 2.0), (3.0, 0.3, 0.0)],
)
def test_invalid_reference_parameters_are_rejected(start, end, tau):
    pytest.importorskip("matplotlib")
    namespace = runpy.run_path(str(SCRIPT))
    with pytest.raises(ValueError):
        namespace["synthetic_otp_reference"](
            [1.0, 2.0],
            start=start,
            end=end,
            tau=tau,
        )
