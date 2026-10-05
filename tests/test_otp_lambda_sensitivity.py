import math
from pathlib import Path

from manchu_ocr.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    0.0: "svtr_dab_otp_lambda_0.yaml",
    0.025: "svtr_dab_otp_lambda_0p025.yaml",
    0.05: "svtr_dab_otp_lambda_0p05.yaml",
    0.1: "svtr_dab_otp_lambda_0p1.yaml",
    0.2: "svtr_dab_otp_lambda_0p2.yaml",
    0.5: "svtr_dab_otp_lambda_0p5.yaml",
}
FIXED_LOSS_KEYS = {
    "blank_idx",
    "zero_infinity",
    "use_alignment_loss",
    "lambda_align",
    "lambda_align_final",
    "lambda_align_decay_start_epoch",
    "lambda_align_decay_end_epoch",
    "align_label_smoothing",
    "lambda_align_activate_epoch",
    "lambda_align_warmup_epochs",
}


def rec_config(filename: str) -> dict:
    return load_yaml(PROJECT_ROOT / "configs" / "recognition" / filename)


def test_lambda_sweep_is_complete_and_controlled():
    ours = rec_config("svtr_official_dab_lortho.yaml")

    for lambda_value, filename in CONFIGS.items():
        cfg = rec_config(filename)
        assert math.isclose(float(cfg["ablation"]["value"]), lambda_value)
        assert cfg["ablation"]["study"] == "otp_lambda_sensitivity"
        assert cfg["model"] == ours["model"]
        assert cfg["data"] == ours["data"]
        assert cfg["decode"] == ours["decode"]
        assert cfg["train"] == ours["train"]
        assert {
            key: cfg["loss"][key] for key in FIXED_LOSS_KEYS
        } == {
            key: ours["loss"][key] for key in FIXED_LOSS_KEYS
        }


def test_lambda_zero_disables_otp_but_keeps_dab():
    cfg = rec_config(CONFIGS[0.0])
    assert cfg["model"]["use_diacritic_branch"] is True
    assert cfg["loss"]["use_orthographic_loss"] is False
    assert cfg["ablation"]["otp_enabled"] is False
    assert "lambda_ortho" not in cfg["loss"]
    assert "transition_matrix" not in cfg["loss"]


def test_positive_lambda_is_constant_from_first_epoch():
    for lambda_value in [0.025, 0.05, 0.1, 0.2, 0.5]:
        cfg = rec_config(CONFIGS[lambda_value])
        loss = cfg["loss"]
        assert loss["use_orthographic_loss"] is True
        assert math.isclose(float(loss["lambda_ortho"]), lambda_value)
        assert not any(key.startswith("lambda_ortho_") for key in loss)
        assert "transition_matrix" in loss


def test_validation_and_test_metric_logic_is_unified_and_strict():
    for filename in CONFIGS.values():
        cfg = rec_config(filename)
        assert "metrics" not in cfg
        assert cfg["train"]["best_metric"] == "word_accuracy"
        assert cfg["train"]["validate_during_training"] is True
        assert int(cfg["train"]["val_interval"]) == 50


def test_experiment_plan_and_runner_cover_all_six_configs():
    plan = load_yaml(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "rec_otp_lambda_sensitivity.yaml"
    )
    planned = {
        Path(item["config"]).name: float(item["lambda"])
        for item in plan["experiments"]
    }
    assert planned == {filename: value for value, filename in CONFIGS.items()}
    assert plan["validation_policy"]["every_epoch"] is False
    assert plan["validation_policy"]["interval_epochs"] == 50
    assert plan["validation_policy"]["final_epoch_always"] is True
    assert plan["validation_policy"]["metric_protocol"] == "strict_for_all_lambda_values"
    assert plan["validation_policy"]["tolerance_applied"] is False


def test_main_otp_weight_is_fixed_at_point_one():
    loss = rec_config("svtr_official_dab_lortho.yaml")["loss"]
    assert math.isclose(float(loss["lambda_ortho"]), 0.1)
    assert not any(key.startswith("lambda_ortho_") for key in loss)

    runner = (
        PROJECT_ROOT / "scripts" / "run_rec_otp_lambda_sensitivity.sh"
    ).read_text(encoding="utf-8")
    for filename in CONFIGS.values():
        assert f"configs/recognition/{filename}" in runner
    assert "for split in val test" in runner
    assert "best.pth" in runner
