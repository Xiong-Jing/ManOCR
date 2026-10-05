"""Utilities for zero-shot OCR evaluation through hosted APIs."""

from .zero_shot import (
    RecognitionSample,
    aggregate_strict_metrics,
    deterministic_subset,
    load_recognition_manifest,
    normalize_romanized_manchu,
)

__all__ = [
    "RecognitionSample",
    "aggregate_strict_metrics",
    "deterministic_subset",
    "load_recognition_manifest",
    "normalize_romanized_manchu",
]
