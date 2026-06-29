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


def first_character_matches(pred: str, label: str) -> bool:
    if len(label) == 0:
        return pred == label

    if len(pred) == 0:
        return False

    return pred[0] == label[0]


def second_character_matches(pred: str, label: str) -> bool:
    if len(label) < 2:
        return True

    if len(pred) < 2:
        return False

    return pred[1] == label[1]


def last_character_matches(pred: str, label: str) -> bool:
    if len(label) == 0:
        return pred == label

    if len(pred) == 0:
        return False

    return pred[-1] == label[-1]


def is_relaxed_word_correct(
    pred: str,
    label: str,
    max_edit_distance: int = 0,
    require_first_char_match: bool = False,
    require_second_char_match: bool = False,
    require_last_char_match: bool = False,
) -> bool:
    dist = levenshtein_distance(pred, label)

    if dist > max(0, int(max_edit_distance)):
        return False

    if require_first_char_match and not first_character_matches(pred, label):
        return False

    if require_second_char_match and not second_character_matches(pred, label):
        return False

    if require_last_char_match and not last_character_matches(pred, label):
        return False

    return True


def compute_recognition_metrics(
    preds: List[str],
    labels: List[str],
    word_correct_max_edit_distance: int = 0,
    char_correct_max_edit_distance: int = 0,
    word_correct_require_first_char_match: bool = False,
    word_correct_require_second_char_match: bool = False,
    word_correct_require_last_char_match: bool = False,
) -> Dict[str, float]:
    """
    Compute word accuracy, character accuracy, and CER.
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
            "word_accuracy_edit_distance": int(word_correct_max_edit_distance),
            "character_accuracy_edit_distance": int(char_correct_max_edit_distance),
            "word_accuracy_require_first_char_match": bool(word_correct_require_first_char_match),
            "word_accuracy_require_second_char_match": bool(word_correct_require_second_char_match),
            "word_accuracy_require_last_char_match": bool(word_correct_require_last_char_match),
        }

    correct_words = 0
    exact_correct_words = 0
    total_edit_distance = 0
    total_relaxed_edit_distance = 0
    total_chars = 0
    word_correct_max_edit_distance = max(0, int(word_correct_max_edit_distance))
    char_correct_max_edit_distance = max(0, int(char_correct_max_edit_distance))

    for pred, label in zip(preds, labels):
        dist = levenshtein_distance(pred, label)
        if dist == 0:
            exact_correct_words += 1

        if (
            dist <= word_correct_max_edit_distance
            and (
                not word_correct_require_first_char_match
                or first_character_matches(pred, label)
            )
            and (
                not word_correct_require_second_char_match
                or second_character_matches(pred, label)
            )
            and (
                not word_correct_require_last_char_match
                or last_character_matches(pred, label)
            )
        ):
            correct_words += 1

        total_edit_distance += dist
        total_relaxed_edit_distance += max(0, dist - char_correct_max_edit_distance)
        total_chars += len(label)

    word_accuracy = correct_words / total
    exact_word_accuracy = exact_correct_words / total
    strict_cer = total_edit_distance / max(total_chars, 1)
    cer = total_relaxed_edit_distance / max(total_chars, 1)
    character_accuracy = 1.0 - cer
    strict_character_accuracy = 1.0 - strict_cer

    return {
        "word_accuracy": float(word_accuracy),
        "exact_word_accuracy": float(exact_word_accuracy),
        "character_accuracy": float(character_accuracy),
        "cer": float(cer),
        "strict_character_accuracy": float(strict_character_accuracy),
        "strict_cer": float(strict_cer),
        "edit_distance": float(total_relaxed_edit_distance / total),
        "strict_edit_distance": float(total_edit_distance / total),
        "word_accuracy_edit_distance": int(word_correct_max_edit_distance),
        "character_accuracy_edit_distance": int(char_correct_max_edit_distance),
        "word_accuracy_require_first_char_match": bool(word_correct_require_first_char_match),
        "word_accuracy_require_second_char_match": bool(word_correct_require_second_char_match),
        "word_accuracy_require_last_char_match": bool(word_correct_require_last_char_match),
    }
