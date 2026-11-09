"""Dataset size and deterministic partition contract for page detection."""
from pathlib import Path

from manchu_ocr.utils.config import load_yaml


def resolve_detection_data(paths_config):
    data = dict(paths_config["detection_data"])
    if data.get("dataset_config"):
        # Dataset metadata is authoritative; machine-specific paths remain local.
        profile = load_yaml(data["dataset_config"])
        data.update(profile)
    return data


def validate_detection_inventory(items, config):
    images = [str(Path(image).resolve()) for image, _ in items]
    if len(images) != len(set(images)):
        raise ValueError("Multiple detection annotations refer to the same source image")
    expected = config.get("expected_valid_pages")
    if expected is not None and len(items) != int(expected):
        raise ValueError(f"Detection dataset requires {expected} valid pages; found {len(items)}")


def validate_detection_manifests(config):
    from manchu_ocr.data.page_splits import read_manifest
    groups = {split: read_manifest(config[f"{split}_list"]) for split in ("train", "val", "test")}
    counts = config.get("split", {}).get("counts")
    if counts:
        for split, items in groups.items():
            if len(items) != int(counts[split]):
                raise ValueError(f"{split}: expected {counts[split]} detection pages, found {len(items)}")
    all_items = [item for items in groups.values() for item in items]
    validate_detection_inventory(all_items, config)
    return {split: len(items) for split, items in groups.items()}
