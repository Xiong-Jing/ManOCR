from pathlib import Path

from manchu_ocr.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_runner_namespace():
    import runpy

    return runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_heldout_tests.py"))


def test_heldout_plan_covers_every_model_config():
    namespace = load_runner_namespace()
    configured_model_paths = namespace["configured_model_paths"]
    plan = load_yaml(PROJECT_ROOT / "configs" / "experiments" / "heldout_all_models.yaml")

    for task in ("recognition", "detection"):
        planned = {path.resolve() for path in configured_model_paths(plan, task)}
        actual = {
            path.resolve()
            for path in (PROJECT_ROOT / "configs" / task).glob("*.yaml")
        }
        assert planned == actual


def test_validation_and_test_commands_only_change_split():
    namespace = load_runner_namespace()
    ModelEvaluation = namespace["ModelEvaluation"]
    build_evaluation_command = namespace["build_evaluation_command"]

    evaluation = ModelEvaluation(
        task="recognition",
        experiment_name="crnn_baseline",
        model_architecture="CRNNRecognizer",
        config_path=Path("configs/recognition/crnn_baseline.yaml"),
        evaluator_path=Path("scripts/eval_recognition.py"),
        checkpoint_path=Path("outputs/checkpoints/recognition/crnn_baseline/best.pth"),
        checkpoint_tag="best",
        manifests={
            "validation": Path("/data/val.txt"),
            "test": Path("/data/test.txt"),
        },
        batch_size=256,
        num_workers=8,
        save_predictions=True,
        evaluator_args=(),
    )

    validation = build_evaluation_command(
        evaluation,
        split="validation",
        python_bin="python",
    )
    test = build_evaluation_command(
        evaluation,
        split="test",
        python_bin="python",
    )

    validation_split_index = validation.index("--split") + 1
    test_split_index = test.index("--split") + 1
    assert validation[validation_split_index] == "val"
    assert test[test_split_index] == "test"

    validation_without_split = validation.copy()
    test_without_split = test.copy()
    validation_without_split[validation_split_index] = "<split>"
    test_without_split[test_split_index] = "<split>"
    assert validation_without_split == test_without_split
