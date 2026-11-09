"""Explicit source-page provenance; never infer page identity from crop names."""
from __future__ import annotations

import csv
import random
from pathlib import Path


def path_key(value: str) -> str:
    return str(value).replace("\\", "/")


def load_page_mapping(path: str | Path) -> dict[str, str]:
    """CSV columns: image_path, source_page_id (for pages AND their word crops)."""
    mapping = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image_path", "source_page_id"} <= set(reader.fieldnames or []):
            raise ValueError("Mapping requires image_path,source_page_id columns")
        for row in reader:
            image, page = path_key(row["image_path"].strip()), row["source_page_id"].strip()
            if not image or not page or (image in mapping and mapping[image] != page):
                raise ValueError(f"Empty/conflicting page mapping: {row}")
            mapping[image] = page
    if not mapping:
        raise ValueError("Page mapping is empty")
    return mapping


def page_ids(samples, mapping):
    missing = [path for path, _ in samples if path_key(path) not in mapping]
    if missing:
        raise ValueError(f"Missing source_page_id for {len(missing)} images: {missing[:5]}")
    return {mapping[path_key(path)] for path, _ in samples}


def read_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Malformed manifest row in {path}")
        rows.append(tuple(parts))
    return rows


def fixed_page_assignments(manifests, mapping):
    assignments = {}
    for split, path in manifests.items():
        for page in page_ids(read_manifest(path), mapping):
            if page in assignments and assignments[page] != split:
                raise ValueError(f"Source-page leakage: {page} in {assignments[page]} and {split}")
            assignments[page] = split
    return assignments


def split_by_page(samples, mapping, train_ratio=0.8, val_ratio=0.1, seed=42, assignments=None):
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected positive train/val/test proportions")
    assignments = dict(assignments or {})
    if any(split not in {"train", "val", "test"} for split in assignments.values()):
        raise ValueError("Unknown split assignment")
    remaining = sorted(page_ids(samples, mapping) - assignments.keys())
    random.Random(seed).shuffle(remaining)
    n_train, n_val = int(len(remaining) * train_ratio), int(len(remaining) * val_ratio)
    for i, page in enumerate(remaining):
        assignments[page] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    groups = {s: [] for s in ("train", "val", "test")}
    for sample in samples:
        groups[assignments[mapping[path_key(sample[0])]]].append(sample)
    return groups["train"], groups["val"], groups["test"]


def audit_page_splits(sections, mapping):
    """Enforce same split across tasks, including E2E validation/test exclusions."""
    seen, report = {}, {}
    for task, manifests in sections.items():
        report[task] = {}
        for split, path in manifests.items():
            samples = read_manifest(path)
            if not samples:
                raise ValueError(f"Empty {task}/{split} manifest")
            pages = page_ids(samples, mapping)
            report[task][split] = {"samples": len(samples), "pages": len(pages)}
            for page in pages:
                if page in seen and seen[page][0] != split:
                    raise ValueError(f"Source-page leakage: {page}: {seen[page]} versus {(split, task)}")
                seen[page] = (split, task)
    return report
