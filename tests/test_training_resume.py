"""Check the production checkpoint/resume blocks without running OCR training."""

import ast
import json
import os
import re
import runpy
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from manchu_ocr.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ORDER = PROJECT_ROOT / "docs" / "supplementary_experiment_run_order.md"
QUEUE_GROUPS = (
    (1, "detection", (
        "dbnetpp_official_baseline", "dbnetpp_strip_pooling",
        "dbnetpp_coordinate_attention", "dbnetpp_horizontal_strip",
        "dbnetpp_vertical_strip", "dbnetpp_vsaa",
    )),
    (2, "recognition", ("svtrv2_nrtr_baseline",)),
    (3, "recognition", (
        "svtr_dab_otp_lambda_0p1", "svtr_dab_otp_lambda_0",
        "svtr_dab_otp_lambda_0p025", "svtr_dab_otp_lambda_0p05",
        "svtr_dab_otp_lambda_0p2", "svtr_dab_otp_lambda_0p5",
    )),
    (4, "recognition", (
        "svtrv2_baseline", "dcm_baseline", "crnn_baseline",
        "parseq_baseline", "abinet_baseline",
    )),
    (5, "detection", ("ppocrv5_det_baseline", "hisam_baseline")),
    (6, "recognition", ("svtr_official_dab_lortho",)),
)
COMPONENT_CASES = (
    ("detection", False), ("recognition", False), ("recognition", True),
)


def _trainer_source(task):
    return PROJECT_ROOT / "scripts" / f"train_{task}.py"


def _compile_nodes(task, nodes):
    # Preserve real production logic, but avoid importing all OCR dependencies.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    return compile(ast.fix_missing_locations(module), str(_trainer_source(task)), "exec")


def _save_function(task, torch):
    tree = ast.parse(_trainer_source(task).read_text(encoding="utf-8-sig"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint"
    )
    namespace = {"torch": torch, "Path": Path}
    exec(_compile_nodes(task, [function]), namespace)
    return namespace["save_checkpoint"]


def _resume_code(task):
    tree = ast.parse(_trainer_source(task).read_text(encoding="utf-8-sig"))
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    index = next(
        index for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "resume_path"
                for target in node.targets)
    )
    nodes = main.body[index:index + 3]
    assert isinstance(nodes[1], ast.If)
    assert isinstance(nodes[2], ast.If)
    assert ast.unparse(nodes[2].test) == "resume_path is not None"
    return _compile_nodes(task, nodes)


def _components(torch, use_ema):
    model = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    lr_lambda = lambda step: 1.0 / (step + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0, enabled=True)
    ema_model = torch.nn.Linear(2, 1, bias=False) if use_ema else None
    if ema_model is not None:
        torch.nn.init.constant_(ema_model.weight, 2.0)
    return model, optimizer, scheduler, scaler, ema_model, lr_lambda


def _resume_namespace(torch, tmp_path, *, resume=None, auto_resume=False, use_ema=False):
    model, optimizer, scheduler, scaler, ema_model, lr_lambda = _components(torch, use_ema)
    return {
        "torch": torch,
        "Path": Path,
        "args": SimpleNamespace(resume=resume, auto_resume=auto_resume),
        "ckpt_dir": tmp_path / "checkpoints",
        "metrics_dir": tmp_path / "metrics",
        "device": torch.device("cpu"),
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "ema_model": ema_model,
        "lr_lambda": lr_lambda,
        "base_lr": 0.01,
        "steps_per_epoch": 3,
        "start_epoch": 1,
        "history": [],
        "best_val_loss": float("inf"),
        "best_val_fmeasure": -1.0,
        "best_word_accuracy": -1.0,
        "best_metric_value": -1.0,
        "logger": Mock(),
    }


@pytest.mark.parametrize("task,use_ema", COMPONENT_CASES)
def test_production_checkpoint_round_trip_restores_next_epoch(task, use_ema, tmp_path):
    torch = pytest.importorskip("torch")
    model, optimizer, scheduler, scaler, ema_model, _ = _components(torch, use_ema)
    for _ in range(4 * 3):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model(torch.ones(1, 2)).sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    expected_weights = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "checkpoints" / "last.pth"
    kwargs = dict(
        path=path, model=model, optimizer=optimizer, epoch=4, cfg={},
        scheduler=scheduler, scaler=scaler,
    )
    if task == "detection":
        kwargs.update(best_val_loss=0.2, best_val_fmeasure=0.8)
    else:
        kwargs.update(
            best_word_accuracy=0.8, best_metric_name="word_accuracy",
            best_metric_value=0.8, ema_model=ema_model,
        )
    _save_function(task, torch)(**kwargs)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert {"epoch", "model", "optimizer", "scheduler", "scaler", "config"} <= checkpoint.keys()

    namespace = _resume_namespace(torch, tmp_path, resume=str(path), use_ema=use_ema)
    namespace["metrics_dir"].mkdir()
    history = [{"epoch": epoch, "val_fmeasure": epoch / 10} for epoch in range(1, 6)]
    (namespace["metrics_dir"] / "metrics.json").write_text(json.dumps(history), encoding="utf-8")
    exec(_resume_code(task), namespace)

    assert namespace["completed_epoch"] == 4
    assert namespace["start_epoch"] == 5
    assert namespace["history"] == history[:4]
    for key, expected in expected_weights.items():
        torch.testing.assert_close(namespace["model"].state_dict()[key], expected)
    assert namespace["optimizer"].state
    assert namespace["optimizer"].param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for restored, expected in zip(
        namespace["optimizer"].state.values(), optimizer.state.values()
    ):
        for key in expected:
            torch.testing.assert_close(restored[key], expected[key])
    assert namespace["scheduler"].state_dict() == scheduler.state_dict()
    assert namespace["scaler"].state_dict() == scaler.state_dict()
    if task == "detection":
        assert namespace["best_val_loss"] == 0.2
        assert namespace["best_val_fmeasure"] == 0.8
    else:
        assert "train_model" in checkpoint
        assert namespace["best_word_accuracy"] == 0.8
        assert namespace["best_metric_value"] == 0.8
        if use_ema:
            torch.testing.assert_close(namespace["ema_model"].weight, ema_model.weight)
            assert not torch.equal(namespace["model"].weight, namespace["ema_model"].weight)
    assert any(
        "completed_epoch=4, start_epoch=5" in str(call.args[0])
        for call in namespace["logger"].info.call_args_list
    )


@pytest.mark.parametrize("task", ("detection", "recognition"))
def test_explicit_missing_checkpoint_fails_instead_of_restarting(task, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    namespace = _resume_namespace(torch, tmp_path, resume=str(tmp_path / "missing.pth"))
    load = Mock(side_effect=AssertionError("missing path must not be loaded"))
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        exec(_resume_code(task), namespace)
    load.assert_not_called()


def test_recognition_resume_keeps_new_validation_frequency(tmp_path):
    torch = pytest.importorskip("torch")
    model, optimizer, scheduler, scaler, ema_model, _ = _components(torch, True)
    path = tmp_path / "checkpoints" / "last.pth"
    _save_function("recognition", torch)(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=6,
        best_word_accuracy=0.8,
        cfg={"train": {"val_interval": 1, "rerank_val_interval": 1}},
        best_metric_name="ctc_rerank_word_accuracy",
        best_metric_value=0.8,
        ema_model=ema_model,
        scheduler=scheduler,
        scaler=scaler,
    )
    namespace = _resume_namespace(torch, tmp_path, resume=str(path), use_ema=True)
    new_config = {"train": {"val_interval": 50, "rerank_val_interval": 50}}
    namespace["cfg"] = new_config
    exec(_resume_code("recognition"), namespace)
    assert namespace["start_epoch"] == 7
    assert namespace["cfg"] == new_config
    assert namespace["best_metric_value"] == 0.8


@pytest.mark.parametrize("task", ("detection", "recognition"))
def test_auto_resume_without_last_has_no_recovery_guarantee(task, tmp_path):
    torch = pytest.importorskip("torch")
    namespace = _resume_namespace(torch, tmp_path, auto_resume=True)
    exec(_resume_code(task), namespace)
    assert namespace["resume_path"] is None
    assert namespace["start_epoch"] == 1


@pytest.mark.parametrize("task", ("detection", "recognition"))
def test_invalid_checkpoint_without_model_is_rejected(task, tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "invalid.pth"
    torch.save({"epoch": 4}, path)
    namespace = _resume_namespace(torch, tmp_path, resume=str(path))
    with pytest.raises(KeyError, match="does not contain key 'model'"):
        exec(_resume_code(task), namespace)


@pytest.mark.parametrize("number,task,names", QUEUE_GROUPS)
def test_all_formal_queue_resume_commands_keep_original_training_settings(number, task, names):
    document = RUN_ORDER.read_text(encoding="utf-8")
    section = document.split(f"### 顺序 {number}：", 1)[1].split("\n### ", 1)[0]
    initial, recovery = section.split(f"#### 顺序 {number} 的断点续跑命令", 1)
    initial_command = re.search(r"```bash\n(.*?)\n```", initial, re.S).group(1)
    recovery_command = re.search(r"```bash\n(.*?)\n```", recovery, re.S).group(1)
    assert f"scripts/train_{task}.py" in recovery_command
    assert "--resume " in recovery_command
    assert f"/checkpoints/{task}/" in recovery_command
    assert "/last.pth\"" in recovery_command
    assert "--auto-resume" not in recovery_command
    assert "tee -a " in recovery_command
    for option in ("batch-size", "num-workers"):
        pattern = rf"--{option}\s+(\S+)"
        assert re.search(pattern, initial_command).group(1) == re.search(pattern, recovery_command).group(1)
    if number == 4:
        assert "crnn_baseline) batch_size=128" in recovery_command
        assert "*) batch_size=64" in recovery_command
    for name in names:
        assert name in initial_command
        assert name in recovery_command
        cfg = load_yaml(PROJECT_ROOT / "configs" / task / f"{name}.yaml")
        assert cfg["experiment"]["name"] == name
        assert cfg["train"]["val_interval"] == (1 if task == "detection" else 50)
        assert cfg["train"]["epochs"] == (100 if task == "detection" else 200)
    trainer_text = _trainer_source(task).read_text(encoding="utf-8")
    assert '"--resume"' in trainer_text
    assert '"--auto-resume"' in trainer_text


def test_run_order_bash_commands_have_valid_shell_syntax():
    bash = shutil.which("bash")
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        ]
        git = shutil.which("git")
        if git:
            for root in Path(git).parents[:2]:
                candidates.extend((root / "bin/bash.exe", root / "usr/bin/bash.exe"))
        bash = next((str(path) for path in candidates if path.is_file()), bash)
    if not bash:
        pytest.skip("Bash is unavailable on this host")
    document = RUN_ORDER.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)\n```", document, re.S)
    result = subprocess.run(
        [bash, "-n"], input="\n\n".join(blocks), text=True,
        encoding="utf-8", capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_documented_e2e_merge_collects_all_eight_results_without_inference(tmp_path, monkeypatch):
    document = RUN_ORDER.read_text(encoding="utf-8").split("### 11.1", 1)[1]
    python_code = re.search(r"python3 - <<'PY'\n(.*?)\nPY", document, re.S).group(1)
    pipelines = [
        SimpleNamespace(pipeline_id=identifier) for identifier in (
            "dbnetpp_svtr", "ours_detector_ours_recognizer",
            "dbnetpp_ours_recognizer", "ours_detector_svtr",
        )
    ]
    plan = {"outputs": {"metrics_subdir": "metrics/e2e_ocr"}}
    validate = Mock(side_effect=lambda path, evaluation, manifest, split: {
        "pipeline": evaluation.pipeline_id, "split": split,
    })
    write = Mock(return_value=(tmp_path / "comparison.json", tmp_path / "comparison.csv"))
    namespace = {
        "load_yaml": lambda path: plan,
        "build_pipeline_evaluations": lambda plan, root, policy: (pipelines, []),
        "resolve_manifests": lambda plan: {
            "validation": tmp_path / "val.txt", "test": tmp_path / "test.txt",
        },
        "result_path_for": lambda root, plan, evaluation, split: (
            root / evaluation.pipeline_id / f"eval_{split}.json"
        ),
        "validate_result": validate,
        "comparison_row": lambda result, path: result,
        "write_comparison": write,
    }
    monkeypatch.setenv("OCR_MANCHU_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(runpy, "run_path", lambda path: namespace)
    exec(compile(python_code, "documented_e2e_merge", "exec"), {})
    assert validate.call_count == 8
    rows, directory, _ = write.call_args.args
    assert len(rows) == 8
    assert {(row["pipeline"], row["split"]) for row in rows} == {
        (pipeline.pipeline_id, split) for pipeline in pipelines
        for split in ("validation", "test")
    }
    assert directory == tmp_path / "metrics/e2e_ocr"
