#!/usr/bin/env python3
"""Score Romanized Manchu recognition predictions with edit-distance metrics.

Input can be a CSV/TSV file containing at least these two columns:

    ground_truth,prediction

Saved JSON prediction arrays are also supported. Their field names can be
selected with ``--ground-truth-column`` and ``--prediction-column``.

The scorer uses one strict protocol for every model: WA is the exact-match
rate, CA is ``(N - S - D) / N``, and CER is ``(S + D + I) / N``. No edit-
distance tolerance or boundary-character exception is applied. By default,
whitespace is treated as formatting and removed, matching this project's
Romanized Manchu data preparation. Use ``--space-policy count`` only when
spaces represent real token boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_MODULE_PATH = (
    PROJECT_ROOT / "src" / "manchu_ocr" / "metrics" / "recognition_metrics.py"
)
_METRICS_MODULE = runpy.run_path(str(METRICS_MODULE_PATH))
levenshtein_error_counts = _METRICS_MODULE["levenshtein_error_counts"]


OUTPUT_COLUMNS = ("model", "split", "S", "D", "I", "N", "CER", "WA", "CA")
DEFAULT_BLANK_TOKENS = ("<blank>", "[blank]", "<ctc_blank>")


@dataclass(frozen=True)
class ScoreSummary:
    substitutions: int
    deletions: int
    insertions: int
    reference_characters: int
    exact_matches: int
    samples: int

    @property
    def edit_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        return self.edit_errors / self.reference_characters

    @property
    def wa(self) -> float:
        return self.exact_matches / self.samples

    @property
    def ca(self) -> float:
        return (
            self.reference_characters - self.substitutions - self.deletions
        ) / self.reference_characters

    @property
    def strict_ca(self) -> float:
        """Backward-compatible alias for the only supported CA definition."""
        return self.ca


def remove_ctc_blank_tokens(text: str, blank_tokens: Sequence[str]) -> str:
    """Remove configured textual CTC blank markers from a prediction."""
    normalized = text
    for token in sorted(set(blank_tokens), key=len, reverse=True):
        if token:
            normalized = re.sub(re.escape(token), "", normalized, flags=re.IGNORECASE)
    return normalized


def normalize_spaces(text: str, space_policy: str) -> str:
    """Apply the declared Romanized Manchu whitespace policy."""
    if space_policy == "ignore":
        return "".join(text.split())

    if space_policy == "count":
        # Leading/trailing formatting whitespace is discarded. Each internal
        # whitespace run becomes one real token-boundary character.
        return " ".join(text.split())

    raise ValueError(f"Unsupported space policy: {space_policy}")


def normalize_pair(
    ground_truth: str,
    prediction: str,
    blank_tokens: Sequence[str],
    space_policy: str,
) -> tuple[str, str]:
    """Remove prediction blanks, then apply the same space policy to both sides."""
    prediction_without_blanks = remove_ctc_blank_tokens(prediction, blank_tokens)
    return (
        normalize_spaces(ground_truth, space_policy),
        normalize_spaces(prediction_without_blanks, space_policy),
    )


def score_pairs(
    pairs: Iterable[tuple[str, str]],
    blank_tokens: Sequence[str] = DEFAULT_BLANK_TOKENS,
    space_policy: str = "ignore",
) -> ScoreSummary:
    """Compute strict per-sample alignments and corpus-level scores."""
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_reference_characters = 0
    exact_matches = 0
    samples = 0

    for ground_truth, prediction in pairs:
        reference, hypothesis = normalize_pair(
            ground_truth=ground_truth,
            prediction=prediction,
            blank_tokens=blank_tokens,
            space_policy=space_policy,
        )
        substitutions, deletions, insertions = levenshtein_error_counts(
            prediction=hypothesis,
            reference=reference,
        )
        sample_edit_distance = substitutions + deletions + insertions

        total_substitutions += substitutions
        total_deletions += deletions
        total_insertions += insertions
        total_reference_characters += len(reference)
        exact_matches += int(sample_edit_distance == 0)
        samples += 1

    if samples == 0:
        raise ValueError("Input contains no data rows")

    if total_reference_characters == 0:
        raise ValueError("N is zero: all normalized ground_truth values are empty")

    return ScoreSummary(
        substitutions=total_substitutions,
        deletions=total_deletions,
        insertions=total_insertions,
        reference_characters=total_reference_characters,
        exact_matches=exact_matches,
        samples=samples,
    )


def resolve_delimiter(input_path: Path, delimiter_name: str) -> str:
    if delimiter_name == "comma":
        return ","
    if delimiter_name == "tab":
        return "\t"
    if delimiter_name == "auto":
        return "\t" if input_path.suffix.lower() == ".tsv" else ","
    raise ValueError(f"Unsupported delimiter: {delimiter_name}")


def read_csv_pairs(
    input_path: Path,
    delimiter: str,
    ground_truth_column: str,
    prediction_column: str,
) -> Iterator[tuple[str, str]]:
    """Read prediction pairs from CSV/TSV with header validation."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {input_path}")

        header_map = {name.strip(): name for name in reader.fieldnames if name is not None}
        required_columns = (ground_truth_column, prediction_column)
        missing = [name for name in required_columns if name not in header_map]
        if missing:
            raise ValueError(
                f"Missing required column(s) {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        ground_truth_key = header_map[ground_truth_column]
        prediction_key = header_map[prediction_column]

        for row_number, row in enumerate(reader, start=2):
            ground_truth = row.get(ground_truth_key)
            prediction = row.get(prediction_key)
            if ground_truth is None or prediction is None:
                raise ValueError(f"Malformed input row {row_number}: {row}")
            yield ground_truth, prediction


def read_json_pairs(
    input_path: Path,
    ground_truth_column: str,
    prediction_column: str,
) -> Iterator[tuple[str, str]]:
    """Read prediction pairs from a JSON array or a top-level records array."""
    with input_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "records" in data:
        data = data["records"]

    if not isinstance(data, list):
        raise ValueError("JSON input must be an array or contain a 'records' array")

    for row_number, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"JSON record {row_number} is not an object")
        if ground_truth_column not in row or prediction_column not in row:
            raise ValueError(
                f"JSON record {row_number} must contain {ground_truth_column!r} "
                f"and {prediction_column!r}"
            )
        ground_truth = row[ground_truth_column]
        prediction = row[prediction_column]
        if ground_truth is None or prediction is None:
            raise ValueError(f"JSON record {row_number} contains a null text value")
        yield str(ground_truth), str(prediction)


def read_pairs(
    input_path: Path,
    delimiter: str,
    ground_truth_column: str = "ground_truth",
    prediction_column: str = "prediction",
) -> Iterator[tuple[str, str]]:
    """Read CSV, TSV, or JSON prediction pairs."""
    if input_path.suffix.lower() == ".json":
        yield from read_json_pairs(
            input_path=input_path,
            ground_truth_column=ground_truth_column,
            prediction_column=prediction_column,
        )
        return

    yield from read_csv_pairs(
        input_path=input_path,
        delimiter=delimiter,
        ground_truth_column=ground_truth_column,
        prediction_column=prediction_column,
    )


def write_summary(
    output_path: Path,
    model: str,
    split: str,
    summary: ScoreSummary,
    append: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": model,
        "split": split,
        "S": summary.substitutions,
        "D": summary.deletions,
        "I": summary.insertions,
        "N": summary.reference_characters,
        "CER": f"{summary.cer:.10f}",
        "WA": f"{summary.wa:.10f}",
        "CA": f"{summary.ca:.10f}",
    }

    write_header = True
    mode = "w"
    if append and output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_header = csv.DictReader(handle).fieldnames
        if tuple(existing_header or ()) != OUTPUT_COLUMNS:
            raise ValueError(
                f"Cannot append to {output_path}: unexpected header {existing_header}"
            )
        write_header = False
        mode = "a"

    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate standard edit-distance recognition scores from "
            "ground_truth and prediction columns."
        )
    )
    parser.add_argument("input", type=Path, help="Input CSV, TSV, or JSON file")
    parser.add_argument("--model", required=True, help="Model name written to output")
    parser.add_argument("--split", required=True, help="Dataset split written to output")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recognition_scores.csv"),
        help="Output CSV path (default: recognition_scores.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append one result row to an existing output CSV",
    )
    parser.add_argument(
        "--delimiter",
        choices=("auto", "comma", "tab"),
        default="auto",
        help="Input delimiter; auto uses tab for .tsv and comma otherwise",
    )
    parser.add_argument(
        "--space-policy",
        choices=("ignore", "count"),
        default="ignore",
        help=(
            "ignore removes all whitespace (project default); count treats each "
            "normalized internal space as one real token-boundary character"
        ),
    )
    parser.add_argument(
        "--blank-token",
        action="append",
        default=None,
        help=(
            "Textual CTC blank marker removed from prediction; repeat for multiple "
            "markers. Defaults: <blank>, [blank], <ctc_blank>"
        ),
    )
    parser.add_argument(
        "--ground-truth-column",
        default="ground_truth",
        help=(
            "Ground-truth field name (default: ground_truth). Use 'label' for "
            "this project's saved prediction JSON files"
        ),
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
        help="Prediction field name (default: prediction)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = args.model.strip()
    split = args.split.strip()
    if not model:
        raise ValueError("--model must not be empty")
    if not split:
        raise ValueError("--split must not be empty")

    blank_tokens = tuple(args.blank_token or DEFAULT_BLANK_TOKENS)
    delimiter = resolve_delimiter(input_path, args.delimiter)
    summary = score_pairs(
        read_pairs(
            input_path,
            delimiter,
            ground_truth_column=args.ground_truth_column,
            prediction_column=args.prediction_column,
        ),
        blank_tokens=blank_tokens,
        space_policy=args.space_policy,
    )
    write_summary(
        output_path=output_path,
        model=model,
        split=split,
        summary=summary,
        append=args.append,
    )

    space_note = (
        "whitespace ignored as formatting"
        if args.space_policy == "ignore"
        else "internal spaces counted as real token boundaries"
    )
    print(f"Scored {summary.samples} samples ({space_note}).", file=sys.stderr)
    print(
        "Metric protocol: strict exact-match WA; "
        "CA=(N-S-D)/N; CER=(S+D+I)/N.",
        file=sys.stderr,
    )
    print(f"Saved aggregate scores to: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
