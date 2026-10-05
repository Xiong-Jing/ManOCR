from math import isclose

from manchu_ocr.metrics.recognition_metrics import (
    compute_recognition_metrics,
    levenshtein_distance,
    levenshtein_error_counts,
)


def test_levenshtein_error_counts():
    cases = [
        ("abc", "abc", (0, 0, 0)),
        ("axc", "abc", (1, 0, 0)),
        ("ac", "abc", (0, 1, 0)),
        ("abxc", "abc", (0, 0, 1)),
        ("", "abc", (0, 3, 0)),
        ("abc", "", (0, 0, 3)),
    ]

    for prediction, reference, expected_counts in cases:
        counts = levenshtein_error_counts(
            prediction=prediction,
            reference=reference,
        )

        assert counts == expected_counts
        assert sum(counts) == levenshtein_distance(prediction, reference)


def test_cer_uses_corpus_level_edit_error_counts():
    metrics = compute_recognition_metrics(
        preds=["ab", "abc"],
        labels=["a", "abcd"],
    )

    assert metrics["substitutions"] == 0
    assert metrics["deletions"] == 1
    assert metrics["insertions"] == 1
    assert metrics["reference_characters"] == 5
    assert isclose(metrics["cer"], 2 / 5)
    assert metrics["cer"] == metrics["strict_cer"]


def test_all_samples_use_strict_metrics_without_tolerance():
    metrics = compute_recognition_metrics(
        preds=["axc", "axy"],
        labels=["abc", "abc"],
    )

    assert metrics["substitutions"] == 3
    assert metrics["deletions"] == 0
    assert metrics["insertions"] == 0
    assert metrics["reference_characters"] == 6
    assert isclose(metrics["cer"], 3 / 6)
    assert isclose(metrics["strict_cer"], 3 / 6)
    assert isclose(metrics["word_accuracy"], 0.0)
    assert metrics["word_accuracy"] == metrics["exact_word_accuracy"]
    assert isclose(metrics["character_accuracy"], 1 / 2)
    assert isclose(metrics["strict_character_accuracy"], 1 / 2)
    assert metrics["correct_reference_characters"] == 3


def test_cer_is_not_clamped_when_insertions_exceed_reference_length():
    metrics = compute_recognition_metrics(
        preds=["abcd"],
        labels=["a"],
    )

    assert metrics["insertions"] == 3
    assert isclose(metrics["cer"], 3.0)
    assert isclose(metrics["character_accuracy"], 1.0)
    assert isclose(metrics["strict_character_accuracy"], 1.0)


def test_ca_counts_reference_errors_and_cer_also_counts_insertions():
    metrics = compute_recognition_metrics(
        preds=["axyc"],
        labels=["abc"],
    )

    assert metrics["substitutions"] == 1
    assert metrics["deletions"] == 0
    assert metrics["insertions"] == 1
    assert isclose(metrics["cer"], 2 / 3)
    assert isclose(metrics["character_accuracy"], 2 / 3)
    assert isclose(metrics["strict_character_accuracy"], 2 / 3)
