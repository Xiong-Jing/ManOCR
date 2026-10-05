from __future__ import annotations

import json
import runpy
from pathlib import Path

from manchu_ocr.api_eval.providers import (
    ProviderOutput,
    build_openai_vision_payload,
    extract_openai_message_text,
    extract_paddle_text,
)
from manchu_ocr.api_eval.zero_shot import (
    RecognitionSample,
    aggregate_strict_metrics,
    deterministic_subset,
    length_bin_counts,
    load_recognition_manifest,
    normalize_romanized_manchu,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_normalization_matches_project_label_conventions_without_answer_extraction():
    assert normalize_romanized_manchu("  A ` B\n") == "a’b"
    polluted = normalize_romanized_manchu("```text\nAB\n```")
    assert "textab" in polluted
    assert polluted != "textab"


def test_manifest_and_subset_are_deterministic_and_length_stratified(tmp_path: Path):
    manifest = tmp_path / "val.txt"
    rows = []
    for index in range(30):
        label = "a" * (2 if index < 10 else 6 if index < 20 else 9)
        rows.append(f"/images/{index}.png\t{label}\n")
    manifest.write_text("".join(rows), encoding="utf-8")

    samples = load_recognition_manifest(manifest)
    first = deterministic_subset(samples, limit=9, seed=42, split="validation")
    second = deterministic_subset(samples, limit=9, seed=42, split="validation")

    assert first == second
    assert len({item.sample_id for item in samples}) == 30
    assert length_bin_counts(first) == {
        "short_1_4": 3,
        "medium_5_7": 3,
        "long_8_plus": 3,
    }
    assert [item.source_index for item in first] == sorted(
        item.source_index for item in first
    )


def test_openai_payload_contains_only_image_and_frozen_prompt():
    payload = build_openai_vision_payload(
        model_id="qwen2.5-vl-7b-instruct",
        prompt="fixed zero-shot prompt",
        image_uri="data:image/png;base64,AAAA",
        generation={"temperature": 0.0, "max_tokens": 64},
    )
    serialized = json.dumps(payload)

    assert "fixed zero-shot prompt" in serialized
    assert "data:image/png;base64,AAAA" in serialized
    assert "ground_truth" not in serialized
    assert "label" not in serialized


def test_openai_and_paddle_response_extraction():
    assert extract_openai_message_text(
        {"choices": [{"message": {"content": [{"type": "text", "text": "morin"}]}}]}
    ) == "morin"

    class Page:
        markdown_text = "morin"

    class Result:
        pages = [Page()]
        job_id = "job-1"

    text, metadata = extract_paddle_text(Result())
    assert text == "morin"
    assert metadata == {"job_id": "job-1", "pages": 1}


def test_common_lmm_metrics_are_strict_edit_distance_metrics():
    metrics = aggregate_strict_metrics(["abc", "de"], ["axc", "def"])

    assert metrics["S"] == 1
    assert metrics["D"] == 0
    assert metrics["I"] == 1
    assert metrics["N"] == 5
    assert metrics["CER"] == 2 / 5
    assert metrics["WA"] == 0.0
    assert metrics["CA"] == 4 / 5


def test_runner_writes_standard_result_and_resumes_without_new_calls(
    tmp_path: Path,
):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"fake-image-a")
    image_b.write_bytes(b"fake-image-b")
    val_manifest = tmp_path / "val.txt"
    test_manifest = tmp_path / "test.txt"
    val_manifest.write_text(
        f"{image_a}\tmorin\n{image_b}\tambi\n",
        encoding="utf-8",
    )
    test_manifest.write_text(f"{image_a}\tmorin\n", encoding="utf-8")
    paths_config = tmp_path / "paths.yaml"
    output_root = tmp_path / "outputs"
    paths_config.write_text(
        "recognition_data:\n"
        f"  val_list: '{val_manifest.as_posix()}'\n"
        f"  test_list: '{test_manifest.as_posix()}'\n"
        "outputs:\n"
        f"  root: '{output_root.as_posix()}'\n",
        encoding="utf-8",
    )
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "experiment:\n"
        "  name: lmm_test\n"
        "  task: recognition\n"
        "  protocol_version: test_v1\n"
        f"  paths_config: '{paths_config.as_posix()}'\n"
        "protocol:\n"
        "  default_splits: [validation]\n"
        "  selection_seed: 42\n"
        "  prompt: fixed prompt\n"
        "  generation: {temperature: 0.0, max_tokens: 8}\n"
        "  normalization: {remove_whitespace: true, lowercase: true, normalize_apostrophes: true}\n"
        "  retry: {max_attempts: 1, max_errors_before_stop: 1}\n"
        "  progress_every: 1\n"
        "models:\n"
        "  fake_model:\n"
        "    display_name: Fake API OCR\n"
        "    provider: openai_compatible\n"
        "    model_id: fake-v1\n"
        "    base_url: https://example.invalid/v1\n"
        "    api_key_env: UNUSED_KEY\n"
        "    accepts_text_prompt: true\n"
        "    workers: 1\n",
        encoding="utf-8",
    )

    namespace = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_lmm_zero_shot.py"))
    calls: list[str] = []

    class FakeProvider:
        def predict(self, image_path: str) -> ProviderOutput:
            calls.append(str(image_path))
            text = "morin" if Path(image_path).name == "a.png" else "ambi"
            return ProviderOutput(text=text, metadata={"usage": {"total_tokens": 1}})

        def close(self) -> None:
            return None

    main = namespace["main"]
    main.__globals__["validate_provider_environment"] = lambda config: []
    main.__globals__["create_provider"] = lambda *args, **kwargs: FakeProvider()
    arguments = [
        "--config",
        str(experiment_config),
        "--splits",
        "validation",
        "--run-tag",
        "unit",
    ]

    assert main(arguments) == 0
    assert len(calls) == 2
    assert main(arguments) == 0
    assert len(calls) == 2

    result_path = (
        output_root
        / "metrics"
        / "lmm_zero_shot"
        / "test_v1"
        / "unit"
        / "fake_model"
        / "eval_val.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["model_name"] == "fake_model"
    assert result["split"] == "validation"
    assert result["checkpoint_path"] == "api://openai_compatible/fake-v1"
    assert result["manifest_path"] == val_manifest.as_posix()
    assert result["metrics"]["WA"] == 1.0
    assert result["metrics"]["CER"] == 0.0
    assert result["protocol"]["zero_shot"] is True
    assert result["protocol"]["training_performed"] is False

    test_arguments = [
        "--config",
        str(experiment_config),
        "--splits",
        "test",
        "--run-tag",
        "unit",
    ]
    assert main(test_arguments) == 0
    assert len(calls) == 3
    summary = json.loads(
        (
            output_root
            / "metrics"
            / "lmm_zero_shot"
            / "test_v1"
            / "unit"
            / "lmm_zero_shot_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert {(row["model"], row["split"]) for row in summary["rows"]} == {
        ("Fake API OCR", "validation"),
        ("Fake API OCR", "test"),
    }
    summary_csv = (
        output_root
        / "metrics"
        / "lmm_zero_shot"
        / "test_v1"
        / "unit"
        / "lmm_zero_shot_summary.csv"
    )
    assert summary_csv.read_text(encoding="utf-8-sig").splitlines()[0].startswith(
        "model,split,S,D,I,N,CER,WA,CA,"
    )
