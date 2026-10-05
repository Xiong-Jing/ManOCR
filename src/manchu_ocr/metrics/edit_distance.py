from manchu_ocr.metrics.recognition_metrics import levenshtein_distance


def character_error_rate(pred: str, label: str) -> float:
    """Compute CER for one prediction-label pair."""
    return levenshtein_distance(pred, label) / max(len(label), 1)


__all__ = ["levenshtein_distance", "character_error_rate"]
