from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from manchu_ocr.metrics.recognition_metrics import (
    compute_recognition_metrics,
    levenshtein_error_counts,
)


_APOSTROPHE_VARIANTS = str.maketrans(
    {
        "‘": "’",
        "`": "’",
        "ʼ": "’",
        "\u02bc": "’",
    }
)


@dataclass(frozen=True)
class RecognitionSample:
    """One immutable word-crop evaluation sample."""

    source_index: int
    image_path: str
    label: str
    sample_id: str


def _sample_id(source_index: int, image_path: str, label: str) -> str:
    material = f"{source_index}\0{image_path}\0{label}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:20]


def load_recognition_manifest(path: str | Path) -> list[RecognitionSample]:
    """Load the project's ``image_path<TAB>label`` recognition manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Recognition manifest not found: {manifest_path}")

    samples: list[RecognitionSample] = []
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid manifest row at {manifest_path}:{line_number}; "
                    "expected image_path<TAB>label"
                )
            image_path, label = (part.strip() for part in parts)
            if not image_path or not label:
                raise ValueError(
                    f"Empty image path or label at {manifest_path}:{line_number}"
                )
            source_index = len(samples)
            samples.append(
                RecognitionSample(
                    source_index=source_index,
                    image_path=image_path,
                    label=label,
                    sample_id=_sample_id(source_index, image_path, label),
                )
            )

    if not samples:
        raise ValueError(f"Recognition manifest contains no samples: {manifest_path}")
    return samples


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _length_bin(label: str) -> str:
    length = len(label)
    if length <= 4:
        return "short_1_4"
    if length <= 7:
        return "medium_5_7"
    return "long_8_plus"


def length_bin_counts(samples: Iterable[RecognitionSample]) -> dict[str, int]:
    counts = {"short_1_4": 0, "medium_5_7": 0, "long_8_plus": 0}
    for sample in samples:
        counts[_length_bin(sample.label)] += 1
    return counts


def _hash_rank(sample: RecognitionSample, *, seed: int, split: str) -> str:
    material = (
        f"{seed}\0{split}\0{sample.source_index}\0"
        f"{sample.image_path}\0{sample.label}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _allocate_strata(group_sizes: Mapping[str, int], limit: int) -> dict[str, int]:
    nonempty = [name for name, size in group_sizes.items() if size > 0]
    allocation = {name: 0 for name in group_sizes}
    if limit <= 0:
        return allocation

    # Preserve every length group when the requested pilot is large enough.
    if limit >= len(nonempty):
        for name in nonempty:
            allocation[name] = 1
        remaining = limit - len(nonempty)
        capacities = {
            name: group_sizes[name] - allocation[name] for name in nonempty
        }
    else:
        remaining = limit
        capacities = {name: group_sizes[name] for name in nonempty}

    while remaining > 0:
        capacity_total = sum(capacities.values())
        if capacity_total <= 0:
            break
        quotas = {
            name: remaining * capacities[name] / capacity_total for name in nonempty
        }
        floors = {
            name: min(capacities[name], int(quotas[name])) for name in nonempty
        }
        assigned = sum(floors.values())
        for name, count in floors.items():
            allocation[name] += count
            capacities[name] -= count
        remaining -= assigned
        if remaining <= 0:
            break

        ranked = sorted(
            nonempty,
            key=lambda name: (
                -(quotas[name] - int(quotas[name])),
                -capacities[name],
                name,
            ),
        )
        made_progress = False
        for name in ranked:
            if remaining <= 0:
                break
            if capacities[name] <= 0:
                continue
            allocation[name] += 1
            capacities[name] -= 1
            remaining -= 1
            made_progress = True
        if not made_progress:
            break

    if sum(allocation.values()) != limit:
        raise RuntimeError("Failed to allocate the requested stratified subset")
    return allocation


def deterministic_subset(
    samples: Sequence[RecognitionSample],
    *,
    limit: int | None,
    seed: int,
    split: str,
) -> list[RecognitionSample]:
    """Select a deterministic, length-stratified pilot subset.

    Full evaluation remains the default when ``limit`` is ``None``. Selected
    samples are returned in source-manifest order so API caches and output rows
    remain easy to audit.
    """
    if limit is None or limit >= len(samples):
        return list(samples)
    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    groups: dict[str, list[RecognitionSample]] = {
        "short_1_4": [],
        "medium_5_7": [],
        "long_8_plus": [],
    }
    for sample in samples:
        groups[_length_bin(sample.label)].append(sample)

    allocation = _allocate_strata(
        {name: len(group) for name, group in groups.items()},
        limit,
    )
    selected: list[RecognitionSample] = []
    for name, group in groups.items():
        ranked = sorted(
            group,
            key=lambda item: _hash_rank(item, seed=seed, split=split),
        )
        selected.extend(ranked[: allocation[name]])
    return sorted(selected, key=lambda item: item.source_index)


def normalize_romanized_manchu(
    text: Any,
    *,
    remove_whitespace: bool = True,
    lowercase: bool = True,
    normalize_apostrophes: bool = True,
) -> str:
    """Apply the same mechanical label rules to every API model output.

    This deliberately does not extract text from explanations, remove Markdown,
    or discard punctuation. Such output-format errors therefore remain visible
    to strict WA/CER scoring.
    """
    normalized = unicodedata.normalize("NFC", "" if text is None else str(text))
    normalized = normalized.strip()
    if remove_whitespace:
        normalized = re.sub(r"\s+", "", normalized)
    if lowercase:
        normalized = normalized.lower()
    if normalize_apostrophes:
        normalized = normalized.translate(_APOSTROPHE_VARIANTS)
    return normalized


def sample_error_counts(reference: str, prediction: str) -> dict[str, int]:
    substitutions, deletions, insertions = levenshtein_error_counts(
        prediction=prediction,
        reference=reference,
    )
    return {
        "S": substitutions,
        "D": deletions,
        "I": insertions,
        "edit_distance": substitutions + deletions + insertions,
        "N": len(reference),
    }


def classify_error_mode(
    *,
    reference: str,
    prediction: str,
    raw_response: str,
    allowed_characters: set[str],
    counts: Mapping[str, int],
) -> str:
    if prediction == reference:
        return "exact"
    if not prediction:
        return "empty_prediction"
    if any(character not in allowed_characters for character in prediction):
        return "unexpected_characters_or_format"
    if "```" in raw_response or len(raw_response.strip().splitlines()) > 1:
        return "multiline_or_markdown_format"

    operations = {name: int(counts[name]) for name in ("S", "D", "I")}
    largest = max(operations.values())
    dominant = sorted(name for name, value in operations.items() if value == largest)
    labels = {"S": "substitution", "D": "deletion", "I": "insertion"}
    if len(dominant) == 1:
        return f"{labels[dominant[0]]}_dominant"
    return "mixed_edit_errors"


def aggregate_strict_metrics(
    references: Sequence[str], predictions: Sequence[str]
) -> dict[str, Any]:
    """Calculate common no-tolerance WA, CA and edit-distance CER."""
    metrics = compute_recognition_metrics(
        preds=list(predictions),
        labels=list(references),
    )
    metrics["num_samples"] = len(references)
    metrics["S"] = int(metrics["substitutions"])
    metrics["D"] = int(metrics["deletions"])
    metrics["I"] = int(metrics["insertions"])
    metrics["N"] = int(metrics["reference_characters"])
    metrics["WA"] = float(metrics["exact_word_accuracy"])
    metrics["CA"] = float(metrics["strict_character_accuracy"])
    metrics["CER"] = float(metrics["strict_cer"])
    return metrics
