"""Training-only, non-blank Manchu transition prior (paper Eqs. 8--10)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

HARMONY_PAIRS = frozenset({("a", "e"), ("e", "a"), ("o", "e"), ("e", "v")})
SPELLING_PAIRS = frozenset({("w", "i"), ("w", "o"), ("w", "u"), ("w", "v")})
RULE_PAIRS = HARMONY_PAIRS | SPELLING_PAIRS


def build_transition_matrix(
    labels: Iterable[str], charset: Sequence[str], smoothing: float = 1.0,
    rho: float = 0.05, epsilon_t: float = 1e-6,
    rule_pairs: Iterable[tuple[str, str]] = RULE_PAIRS,
) -> np.ndarray:
    if not charset or len(set(charset)) != len(charset) or any(len(c) != 1 or c.isspace() for c in charset):
        raise ValueError("charset must contain unique non-blank single characters")
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be positive and finite")
    if not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1]")
    if not np.isfinite(epsilon_t) or not 0 < epsilon_t < 1:
        raise ValueError("epsilon_t must be in (0, 1)")
    indices = {c: i for i, c in enumerate(charset)}
    counts = np.full((len(charset), len(charset)), smoothing, dtype=np.float64)
    for label in labels:
        # No bigrams across whitespace/word boundaries, and no silent OOV loss.
        for word in label.split():
            unknown = set(word) - indices.keys()
            if unknown:
                raise ValueError(f"Training transcription has characters outside charset: {sorted(unknown)}")
            for a, b in zip(word, word[1:]):
                counts[indices[a], indices[b]] += 1
    for a, b in set(rule_pairs):
        if a in indices and b in indices:
            counts[indices[a], indices[b]] *= rho
    matrix = counts / counts.sum(axis=1, keepdims=True)
    matrix = np.maximum(matrix, epsilon_t)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix.astype(np.float32)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_transition_artifact(output: str | Path, matrix: np.ndarray, *, manifest: str | Path,
                              charset_path: str | Path, rho: float, smoothing: float,
                              epsilon_t: float) -> dict:
    output = Path(output)
    if output.suffix != ".npy":
        raise ValueError("Transition artifact must end in .npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, matrix)
    metadata = {
        "schema": "manchu.transition_prior.v2", "source_split": "train",
        "train_manifest": str(manifest), "train_manifest_sha256": sha256_file(manifest),
        "charset_path": str(charset_path), "charset_sha256": sha256_file(charset_path),
        "matrix_sha256": sha256_file(output), "shape": list(matrix.shape),
        "rho": rho, "smoothing": smoothing, "epsilon_t": epsilon_t,
        "harmony_pairs": sorted(HARMONY_PAIRS), "spelling_pairs": sorted(SPELLING_PAIRS),
        "blank_included": False,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
