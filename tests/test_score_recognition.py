import csv
import json
import tempfile
from math import isclose
from pathlib import Path

from manchu_ocr.metrics.recognition_metrics import compute_recognition_metrics
from score_recognition import main, normalize_pair, score_pairs


def test_blank_removal_and_default_space_policy():
    reference, hypothesis = normalize_pair(
        ground_truth="  a b c  ",
        prediction="a<blank> b[BLANK] c",
        blank_tokens=("<blank>", "[blank]"),
        space_policy="ignore",
    )

    assert reference == "abc"
    assert hypothesis == "abc"


def test_score_pairs_aggregates_sdi_and_reference_length():
    summary = score_pairs(
        [
            ("abc", "a<blank>bc"),
            ("abc", "axc"),
            ("abc", "ac"),
            ("abc", "abxc"),
        ]
    )

    assert summary.substitutions == 1
    assert summary.deletions == 1
    assert summary.insertions == 1
    assert summary.reference_characters == 12
    assert summary.exact_matches == 1
    assert summary.samples == 4
    assert isclose(summary.cer, 3 / 12)
    assert isclose(summary.wa, 1 / 4)
    assert isclose(summary.ca, 10 / 12)


def test_space_policy_can_count_real_token_boundaries():
    ignored = score_pairs([("ab cd", "abcd")], space_policy="ignore")
    counted = score_pairs([("ab cd", "abcd")], space_policy="count")

    assert ignored.reference_characters == 4
    assert ignored.deletions == 0
    assert ignored.wa == 1.0

    assert counted.reference_characters == 5
    assert counted.deletions == 1
    assert counted.wa == 0.0


def test_insertions_affect_cer_but_not_reference_character_accuracy():
    summary = score_pairs([("abc", "abxc")])

    assert summary.insertions == 1
    assert isclose(summary.cer, 1 / 3)
    assert isclose(summary.ca, 1.0)


def test_strict_ca_counts_reference_errors_only():
    summary = score_pairs([("abc", "axyc")])

    assert summary.substitutions == 1
    assert summary.insertions == 1
    assert isclose(summary.cer, 2 / 3)
    assert isclose(summary.ca, 2 / 3)
    assert isclose(summary.strict_ca, 2 / 3)


def test_strict_scorer_matches_the_project_metric_function():
    pairs = [
        ("abc", "axc"),
        ("abc", "axy"),
        ("abc", "abxc"),
    ]
    summary = score_pairs(pairs)
    metrics = compute_recognition_metrics(
        preds=[prediction for _, prediction in pairs],
        labels=[reference for reference, _ in pairs],
    )

    assert isclose(summary.wa, metrics["word_accuracy"])
    assert isclose(summary.cer, metrics["cer"])
    assert isclose(summary.ca, metrics["character_accuracy"])
    assert isclose(summary.wa, 0.0)
    assert isclose(summary.cer, 4 / 9)
    assert isclose(summary.ca, 6 / 9)
    assert isclose(summary.strict_ca, 6 / 9)


def test_cli_writes_requested_csv_schema():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "predictions.csv"
        output_path = temp_path / "scores.csv"
        input_path.write_text(
            "ground_truth,prediction\nabc,abc\nabc,axc\n",
            encoding="utf-8",
        )

        exit_code = main(
            [
                str(input_path),
                "--model",
                "example_model",
                "--split",
                "test",
                "--output",
                str(output_path),
            ]
        )

        with output_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        assert exit_code == 0
        assert reader.fieldnames == ["model", "split", "S", "D", "I", "N", "CER", "WA", "CA"]
        assert len(rows) == 1
        assert rows[0]["model"] == "example_model"
        assert rows[0]["split"] == "test"
        assert rows[0]["S"] == "1"
        assert rows[0]["D"] == "0"
        assert rows[0]["I"] == "0"
        assert rows[0]["N"] == "6"
        assert isclose(float(rows[0]["CER"]), 1 / 6)


def test_cli_rescores_saved_prediction_json_without_inference():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "test_predictions.json"
        output_path = temp_path / "scores.csv"
        input_path.write_text(
            json.dumps(
                [
                    {"label": "abc", "prediction": "abxc"},
                    {"label": "abc", "prediction": "axc"},
                ]
            ),
            encoding="utf-8",
        )

        exit_code = main(
            [
                str(input_path),
                "--ground-truth-column",
                "label",
                "--model",
                "example_model",
                "--split",
                "test",
                "--output",
                str(output_path),
            ]
        )

        with output_path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))

        assert exit_code == 0
        assert row["S"] == "1"
        assert row["D"] == "0"
        assert row["I"] == "1"
        assert row["N"] == "6"
        assert isclose(float(row["CER"]), 2 / 6)
        assert isclose(float(row["CA"]), 5 / 6)


def test_cli_uses_strict_metrics_and_appends_rows():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "predictions.csv"
        output_path = temp_path / "scores.csv"
        input_path.write_text(
            "ground_truth,prediction\nabc,axc\nabc,axy\n",
            encoding="utf-8",
        )
        common_args = [
            str(input_path),
            "--model",
            "svtr_official_dab_lortho",
            "--output",
            str(output_path),
        ]

        assert main([*common_args, "--split", "val"]) == 0
        assert main([*common_args, "--split", "test", "--append"]) == 0

        with output_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 2
        assert [row["split"] for row in rows] == ["val", "test"]
        assert all(isclose(float(row["WA"]), 0.0) for row in rows)
        assert all(isclose(float(row["CER"]), 3 / 6) for row in rows)
        assert all(isclose(float(row["CA"]), 3 / 6) for row in rows)
