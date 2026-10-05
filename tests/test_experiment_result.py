from math import isclose

from manchu_ocr.utils.experiment_result import (
    ExperimentTiming,
    build_experiment_result,
    format_experiment_result,
    normalize_split_name,
    split_key,
)


def test_split_names_are_standardized_for_public_results():
    assert normalize_split_name("val") == "validation"
    assert normalize_split_name("validation") == "validation"
    assert normalize_split_name("test") == "test"
    assert split_key("validation") == "val"
    assert split_key("test") == "test"


def test_common_result_contains_required_experiment_fields():
    result = build_experiment_result(
        task="recognition",
        experiment_name="crnn_baseline",
        model_name="crnn_baseline",
        model_architecture="CRNNRecognizer",
        split="val",
        config_path="configs\\recognition\\crnn_baseline.yaml",
        checkpoint_path="outputs\\checkpoints\\recognition\\crnn_baseline\\best.pth",
        manifest_path="/srv/data/recognition/processed/val.txt",
        metrics={"cer": 0.2, "word_accuracy": 0.8},
        timing=ExperimentTiming(
            started_at_utc="2026-01-01T00:00:00+00:00",
            finished_at_utc="2026-01-01T00:00:12+00:00",
            elapsed_seconds=12.3456789,
        ),
        runtime_details={"gpu_name": "NVIDIA GeForce RTX 4090"},
        extra_fields={"checkpoint_epoch": 10},
    )

    assert result["split"] == "validation"
    assert result["split_key"] == "val"
    assert result["checkpoint_path"].endswith("crnn_baseline/best.pth")
    assert result["manifest_path"].endswith("processed/val.txt")
    assert result["metrics"]["cer"] == 0.2
    assert isclose(result["runtime_seconds"], 12.345679)
    assert result["runtime"]["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert result["checkpoint_epoch"] == 10

    log_line = format_experiment_result(result)
    assert "model_name=crnn_baseline" in log_line
    assert "split=validation" in log_line
    assert 'metrics={"cer":0.2,"word_accuracy":0.8}' in log_line
