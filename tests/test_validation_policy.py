import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from manchu_ocr.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _task_configs(task: str):
    return sorted((PROJECT_ROOT / "configs" / task).glob("*.yaml"))


def test_all_recognition_configs_validate_every_50_epochs():
    for config_path in _task_configs("recognition"):
        cfg = load_yaml(config_path)
        train_cfg = cfg["train"]

        assert train_cfg.get("validate_during_training") is True, config_path
        assert int(train_cfg.get("val_interval", 0)) == 50, config_path

        best_metric = str(train_cfg.get("best_metric", "word_accuracy"))
        if best_metric.startswith("ctc_rerank_"):
            assert cfg.get("decode", {}).get("use_ctc_lexicon_rerank") is True, config_path
            assert int(train_cfg.get("rerank_val_interval", 0)) == 50, config_path
        if not cfg.get("decode", {}).get("use_ctc_lexicon_rerank", False):
            assert int(train_cfg.get("rerank_val_interval", 0)) == 0, config_path


def test_all_recognition_configs_use_the_single_strict_metric_protocol():
    for config_path in _task_configs("recognition"):
        cfg = load_yaml(config_path)
        assert "metrics" not in cfg, config_path


def test_all_detection_configs_validate_every_epoch():
    for config_path in _task_configs("detection"):
        cfg = load_yaml(config_path)
        assert int(cfg["train"].get("val_interval", 0)) == 1, config_path


def test_all_detection_configs_use_original_images_and_iou_075():
    for config_path in _task_configs("detection"):
        cfg = load_yaml(config_path)
        assert "eval_degradation" not in cfg, config_path
        assert float(cfg["eval"]["iou_thresh"]) == 0.75, config_path


def test_main_otp_config_uses_fixed_lambda_point_one():
    cfg = load_yaml(
        PROJECT_ROOT
        / "configs"
        / "recognition"
        / "svtr_official_dab_lortho.yaml"
    )
    assert float(cfg["loss"]["lambda_ortho"]) == 0.1
    assert not any(
        key.startswith("lambda_ortho_") for key in cfg["loss"]
    )


def _recognition_main():
    path = PROJECT_ROOT / "scripts" / "train_recognition.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def _schedule_value(name, namespace):
    # Evaluate the actual trainer condition without importing OCR dependencies.
    assignment = next(
        node for node in ast.walk(_recognition_main())
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)
    )
    expression = ast.Expression(body=assignment.value)
    return eval(compile(expression, "train_recognition.py", "eval"), {}, namespace)


@pytest.mark.parametrize(
    "epoch,total_epochs,enabled,debug,expected",
    [
        (1, 200, True, False, False),
        (49, 200, True, False, False),
        (50, 200, True, False, True),
        (51, 200, True, False, False),
        (100, 200, True, False, True),
        (150, 200, True, False, True),
        (199, 200, True, False, False),
        (200, 200, True, False, True),
        (175, 175, True, False, True),
        (1, 1, True, False, True),
        (1, 200, True, True, True),
        (200, 200, False, False, False),
    ],
)
def test_recognition_validation_and_rerank_schedules_stay_aligned(
    epoch, total_epochs, enabled, debug, expected
):
    namespace = {
        "epoch": epoch,
        "epochs": total_epochs,
        "validate_during_training": enabled,
        "val_interval": 50,
        "rerank_val_interval": 50,
        "ctc_lexicon_reranker": object(),
        "args": SimpleNamespace(debug=debug),
    }
    should_validate = _schedule_value("should_validate", namespace)
    assert should_validate is expected
    should_rerank = (
        _schedule_value("should_rerank_validate", namespace)
        if should_validate else False
    )
    assert should_rerank is expected


def test_recognition_without_rerank_keeps_original_decoder():
    namespace = {
        "epoch": 50,
        "epochs": 200,
        "validate_during_training": True,
        "val_interval": 50,
        "rerank_val_interval": 0,
        "ctc_lexicon_reranker": None,
        "args": SimpleNamespace(debug=False),
    }
    assert _schedule_value("should_validate", namespace) is True
    assert _schedule_value("should_rerank_validate", namespace) is False


def test_recognition_still_saves_last_checkpoint_every_epoch():
    epoch_loop = next(
        node for node in ast.walk(_recognition_main())
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == "epoch"
    )
    calls = [
        node.value for node in epoch_loop.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert any(
        isinstance(call.func, ast.Name) and call.func.id == "save_checkpoint"
        and any(keyword.arg == "path"
                and ast.unparse(keyword.value) == "ckpt_dir / 'last.pth'"
                for keyword in call.keywords)
        for call in calls
    )
    assert any(
        isinstance(call.func, ast.Name) and call.func.id == "save_json"
        for call in calls
    )
