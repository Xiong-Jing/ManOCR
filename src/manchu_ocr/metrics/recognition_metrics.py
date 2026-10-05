from typing import Dict, List


def levenshtein_distance(a: str, b: str) -> int:
    """
    Simple edit distance implementation.
    """
    if a == b:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        curr = [i]

        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (ca != cb)

            curr.append(min(insert_cost, delete_cost, replace_cost))

        prev = curr

    return prev[-1]


def levenshtein_error_counts(
    prediction: str,
    reference: str,
) -> tuple[int, int, int]:
    """Return substitution, deletion, and insertion counts.

    The operations describe an optimal Levenshtein alignment that transforms
    ``reference`` into ``prediction``. When multiple optimal alignments exist,
    backtracking prefers substitution, then deletion, then insertion. This
    makes the individual counts deterministic; their sum is the Levenshtein
    distance regardless of the selected optimal alignment.
    """
    reference_length = len(reference)
    prediction_length = len(prediction)

    distances = [
        [0] * (prediction_length + 1)
        for _ in range(reference_length + 1)
    ]

    for reference_index in range(1, reference_length + 1):
        distances[reference_index][0] = reference_index

    for prediction_index in range(1, prediction_length + 1):
        distances[0][prediction_index] = prediction_index

    for reference_index in range(1, reference_length + 1):
        reference_char = reference[reference_index - 1]
        for prediction_index in range(1, prediction_length + 1):
            prediction_char = prediction[prediction_index - 1]
            substitution_cost = int(reference_char != prediction_char)
            distances[reference_index][prediction_index] = min(
                distances[reference_index - 1][prediction_index - 1]
                + substitution_cost,
                distances[reference_index - 1][prediction_index] + 1,
                distances[reference_index][prediction_index - 1] + 1,
            )

    substitutions = 0
    deletions = 0
    insertions = 0
    reference_index = reference_length
    prediction_index = prediction_length

    while reference_index > 0 or prediction_index > 0:
        if (
            reference_index > 0
            and prediction_index > 0
            and reference[reference_index - 1] == prediction[prediction_index - 1]
            and distances[reference_index][prediction_index]
            == distances[reference_index - 1][prediction_index - 1]
        ):
            reference_index -= 1
            prediction_index -= 1
            continue

        if (
            reference_index > 0
            and prediction_index > 0
            and distances[reference_index][prediction_index]
            == distances[reference_index - 1][prediction_index - 1] + 1
        ):
            substitutions += 1
            reference_index -= 1
            prediction_index -= 1
        elif (
            reference_index > 0
            and distances[reference_index][prediction_index]
            == distances[reference_index - 1][prediction_index] + 1
        ):
            deletions += 1
            reference_index -= 1
        elif (
            prediction_index > 0
            and distances[reference_index][prediction_index]
            == distances[reference_index][prediction_index - 1] + 1
        ):
            insertions += 1
            prediction_index -= 1
        else:
            raise RuntimeError("Failed to backtrack Levenshtein alignment")

    return substitutions, deletions, insertions


def compute_recognition_metrics(
    preds: List[str],
    labels: List[str],
) -> Dict[str, float]:
    """Compute the project's strict recognition metrics.

    ``WA`` is the exact-match rate. ``CA`` measures correct reference
    characters and is ``(N - S - D) / N``. ``CER`` is the standard corpus-level
    edit-distance rate ``(S + D + I) / N``. No per-sample tolerance or boundary
    exception is applied.
    """
    if len(preds) != len(labels):
        raise ValueError(f"preds and labels length mismatch: {len(preds)} vs {len(labels)}")

    total = len(labels)

    if total == 0:
        return {
            "word_accuracy": 0.0,
            "exact_word_accuracy": 0.0,
            "character_accuracy": 0.0,
            "cer": 1.0,
            "strict_character_accuracy": 0.0,
            "strict_cer": 1.0,
            "edit_distance": 0.0,
            "strict_edit_distance": 0.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "reference_characters": 0,
            "correct_reference_characters": 0,
        }

    exact_correct_words = 0
    total_edit_distance = 0
    total_chars = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0

    for pred, label in zip(preds, labels):
        substitutions, deletions, insertions = levenshtein_error_counts(
            prediction=pred,
            reference=label,
        )
        dist = substitutions + deletions + insertions
        if dist == 0:
            exact_correct_words += 1

        total_edit_distance += dist
        total_chars += len(label)
        total_substitutions += substitutions
        total_deletions += deletions
        total_insertions += insertions

    exact_word_accuracy = exact_correct_words / total
    cer = total_edit_distance / max(total_chars, 1)
    correct_reference_characters = total_chars - total_substitutions - total_deletions
    character_accuracy = correct_reference_characters / max(total_chars, 1)

    return {
        "word_accuracy": float(exact_word_accuracy),
        "exact_word_accuracy": float(exact_word_accuracy),
        "character_accuracy": float(character_accuracy),
        "cer": float(cer),
        # Compatibility aliases retained for existing result readers. They now
        # refer to the same single strict protocol.
        "strict_character_accuracy": float(character_accuracy),
        "strict_cer": float(cer),
        "edit_distance": float(total_edit_distance / total),
        "strict_edit_distance": float(total_edit_distance / total),
        "substitutions": int(total_substitutions),
        "deletions": int(total_deletions),
        "insertions": int(total_insertions),
        "reference_characters": int(total_chars),
        "correct_reference_characters": int(correct_reference_characters),
    }
