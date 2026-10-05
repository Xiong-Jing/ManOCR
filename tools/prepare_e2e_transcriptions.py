from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manchu_ocr.metrics.detection_metrics import polygon_to_bbox
from manchu_ocr.metrics.e2e_ocr_metrics import parse_page_annotation
from manchu_ocr.pipelines.crop_rectify import crop_polygon_bbox
from manchu_ocr.pipelines.full_ocr_pipeline import FullOCRPipeline
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import save_json, write_txt


VALID_TRANSCRIPTION = re.compile(r"^[a-z’'\-]+$")
SPLIT_KEYS = {
    "train": "train_list",
    "val": "val_list",
    "test": "test_list",
}
CSV_FIELDS = [
    "split",
    "page_index",
    "page_id",
    "image_path",
    "source_annotation_path",
    "source_index",
    "reading_index",
    "column_index",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "review_crop_path",
    "transcription",
    "status",
    "notes",
]


def normalize_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def normalize_transcription(value: str) -> str:
    text = re.sub(r"\s+", "", str(value).strip()).lower()
    return text.replace("‘", "’").replace("`", "’").replace("ʼ", "’")


def load_manifest(path: Path) -> List[Tuple[Path, Path]]:
    samples: List[Tuple[Path, Path]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid manifest line {line_index}: {line!r}")
            samples.append((Path(parts[0]), Path(parts[1])))
    if not samples:
        raise RuntimeError(f"No pages found in manifest: {path}")
    return samples


def resolve_paths(config_path: str | Path) -> Dict[str, Any]:
    config = load_yaml(config_path)
    if "e2e_data" not in config:
        raise KeyError(f"Missing e2e_data section in {config_path}")
    return config


def source_manifests(
    config: Mapping[str, Any],
    splits: Sequence[str],
) -> Dict[str, Path]:
    detection = config["detection_data"]
    return {split: Path(detection[SPLIT_KEYS[split]]) for split in splits}


def output_manifests(
    config: Mapping[str, Any],
    splits: Sequence[str],
) -> Dict[str, Path]:
    e2e = config["e2e_data"]
    return {split: Path(e2e[SPLIT_KEYS[split]]) for split in splits}


def _safe_page_id(split: str, page_index: int, image_path: Path) -> str:
    safe_stem = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", image_path.stem)
    return f"{split}_{page_index:04d}_{safe_stem[:80]}"


def _row_key(annotation_path: str | Path, source_index: int | str) -> Tuple[str, int]:
    return normalize_path(annotation_path), int(source_index)


def read_template(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in CSV_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Template is missing columns {missing}: {path}")
        return [dict(row) for row in reader]


def write_template(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export_template(
    config: Mapping[str, Any],
    splits: Sequence[str],
    template_path: Path,
    crop_dir: Path,
    save_review_crops: bool,
    crop_padding: int,
) -> Dict[str, int]:
    previous: Dict[Tuple[str, int], Dict[str, str]] = {}
    if template_path.is_file():
        previous = {
            _row_key(row["source_annotation_path"], row["source_index"]): row
            for row in read_template(template_path)
        }

    rows: List[Dict[str, Any]] = []
    stats = {"pages": 0, "words": 0, "existing_transcriptions": 0}
    for split, manifest_path in source_manifests(config, splits).items():
        for page_index, (image_path, annotation_path) in enumerate(
            load_manifest(manifest_path),
            start=1,
        ):
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing page image: {image_path}")
            items = parse_page_annotation(annotation_path)
            columns = FullOCRPipeline.group_columns(items)
            page_id = _safe_page_id(split, page_index, image_path)
            page_image = (
                Image.open(image_path).convert("RGB") if save_review_crops else None
            )
            reading_index = 0
            for column in columns:
                for item in column["items"]:
                    source_index = int(item["source_index"])
                    key = _row_key(annotation_path, source_index)
                    old_row = previous.get(key, {})
                    existing_text = normalize_transcription(
                        str(
                            old_row.get("transcription")
                            or item.get("text", "")
                        )
                    )
                    if existing_text:
                        stats["existing_transcriptions"] += 1
                    x1, y1, x2, y2 = polygon_to_bbox(item["box"])
                    crop_path = crop_dir / split / f"{page_id}__word_{source_index:04d}.png"
                    if page_image is not None:
                        crop = crop_polygon_bbox(
                            page_image,
                            item["box"],
                            padding=crop_padding,
                            padding_ratio=0.0,
                        )
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        crop.save(crop_path)
                    rows.append(
                        {
                            "split": split,
                            "page_index": page_index,
                            "page_id": page_id,
                            "image_path": normalize_path(image_path),
                            "source_annotation_path": normalize_path(annotation_path),
                            "source_index": source_index,
                            "reading_index": reading_index,
                            "column_index": column["column_index"],
                            "bbox_x1": round(x1, 4),
                            "bbox_y1": round(y1, 4),
                            "bbox_x2": round(x2, 4),
                            "bbox_y2": round(y2, 4),
                            "review_crop_path": (
                                normalize_path(crop_path) if save_review_crops else ""
                            ),
                            "transcription": existing_text,
                            "status": old_row.get(
                                "status",
                                "done" if existing_text else "todo",
                            ),
                            "notes": old_row.get("notes", ""),
                        }
                    )
                    reading_index += 1
            stats["pages"] += 1
            stats["words"] += len(items)

    write_template(rows, template_path)
    return stats


def _annotation_items(data: Any, path: Path) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("items", "polygons", "shapes"):
            if key in data:
                return data[key]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported annotation structure: {path}")


def validate_template_rows(
    rows: Sequence[Mapping[str, str]],
    splits: Sequence[str],
) -> Dict[Tuple[str, int], str]:
    selected = [row for row in rows if row["split"] in splits]
    mapping: Dict[Tuple[str, int], str] = {}
    errors: List[str] = []
    for row in selected:
        key = _row_key(row["source_annotation_path"], row["source_index"])
        text = normalize_transcription(row["transcription"])
        if not text:
            errors.append(f"missing transcription: {key[0]}#item[{key[1]}]")
            continue
        if not VALID_TRANSCRIPTION.fullmatch(text):
            errors.append(f"invalid Romanized Manchu text {text!r}: {key}")
            continue
        if str(row.get("status", "")).strip().lower() not in {"done", "verified"}:
            errors.append(f"row is not marked done/verified: {key}")
            continue
        if key in mapping:
            errors.append(f"duplicate template row for {key}")
            continue
        mapping[key] = text
    if errors:
        examples = "\n".join(f"  - {error}" for error in errors[:20])
        raise RuntimeError(
            f"Template has {len(errors)} incomplete/invalid row(s). No E2E "
            f"annotation was written. Examples:\n{examples}"
        )
    return mapping


def apply_template(
    config: Mapping[str, Any],
    splits: Sequence[str],
    template_path: Path,
) -> Dict[str, int]:
    rows = read_template(template_path)
    transcription_map = validate_template_rows(rows, splits)
    template_rows = {
        _row_key(row["source_annotation_path"], row["source_index"]): row
        for row in rows
        if row["split"] in splits
    }
    annotation_root = Path(config["e2e_data"]["annotation_dir"])
    prepared: List[Tuple[str, Path, Path, Dict[str, Any]]] = []
    stats = {"pages": 0, "words": 0}

    for split, manifest_path in source_manifests(config, splits).items():
        for image_path, source_annotation in load_manifest(manifest_path):
            with source_annotation.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            source_items = _annotation_items(data, source_annotation)
            parsed_items = parse_page_annotation(source_annotation)
            for parsed in parsed_items:
                source_index = int(parsed["source_index"])
                key = _row_key(source_annotation, source_index)
                if key not in transcription_map:
                    raise RuntimeError(
                        f"Template has no row for {source_annotation}#item[{source_index}]. "
                        "No E2E annotation was written."
                    )
                actual_bbox = polygon_to_bbox(parsed["box"])
                template_bbox = tuple(
                    float(template_rows[key][field])
                    for field in (
                        "bbox_x1",
                        "bbox_y1",
                        "bbox_x2",
                        "bbox_y2",
                    )
                )
                if max(
                    abs(actual - expected)
                    for actual, expected in zip(actual_bbox, template_bbox)
                ) > 0.5:
                    raise RuntimeError(
                        f"Stale template geometry for {key}: "
                        f"annotation={actual_bbox}, template={template_bbox}. "
                        "Re-export and re-check the template; no output was written."
                    )
                source_item = source_items[source_index]
                if not isinstance(source_item, dict):
                    raise TypeError(f"Annotation item is not a mapping: {key}")
                source_item["transcription"] = transcription_map[key]

            if isinstance(data, dict):
                data["e2e_transcription_source"] = normalize_path(template_path)
                data["e2e_split"] = split
            output_annotation = annotation_root / split / source_annotation.name
            prepared.append((split, image_path, output_annotation, data))
            stats["pages"] += 1
            stats["words"] += len(parsed_items)

    output_lines: Dict[str, List[str]] = defaultdict(list)
    for split, image_path, output_annotation, data in prepared:
        save_json(data, output_annotation)
        output_lines[split].append(
            f"{normalize_path(image_path)}\t{normalize_path(output_annotation)}"
        )
    for split, manifest_path in output_manifests(config, splits).items():
        write_txt(output_lines[split], manifest_path)
    return stats


def audit_e2e_manifests(
    config: Mapping[str, Any],
    splits: Sequence[str],
) -> Dict[str, int]:
    stats = {"pages": 0, "words": 0}
    missing: List[str] = []
    for split, manifest_path in output_manifests(config, splits).items():
        for _, annotation_path in load_manifest(manifest_path):
            items = parse_page_annotation(annotation_path)
            stats["pages"] += 1
            stats["words"] += len(items)
            missing.extend(
                f"{annotation_path}#item[{item['source_index']}]"
                for item in items
                if not item["text"]
            )
    if missing:
        raise RuntimeError(
            f"E2E audit found {len(missing)} missing transcription(s): "
            + ", ".join(missing[:10])
        )
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export GT-word review crops/CSV, apply checked transcriptions to "
            "separate E2E annotations, or audit the resulting manifests."
        )
    )
    parser.add_argument(
        "command",
        choices=["export", "apply", "audit"],
    )
    parser.add_argument(
        "--config",
        default="configs/paths/remote_server.yaml",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["val", "test"],
    )
    parser.add_argument("--template", default=None)
    parser.add_argument("--crop-dir", default=None)
    parser.add_argument("--save-review-crops", action="store_true")
    parser.add_argument("--crop-padding", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = resolve_paths(args.config)
    e2e = config["e2e_data"]
    template_path = Path(args.template or e2e["transcription_template"])
    crop_dir = Path(args.crop_dir or e2e["review_crop_dir"])

    if args.command == "export":
        stats = export_template(
            config,
            args.splits,
            template_path,
            crop_dir,
            args.save_review_crops,
            args.crop_padding,
        )
        print(
            f"[OK] template={template_path} pages={stats['pages']} "
            f"words={stats['words']} existing={stats['existing_transcriptions']}"
        )
        if args.save_review_crops:
            print(f"[OK] review_crops={crop_dir}")
        print(
            "Fill every transcription cell, set status=done after review, then "
            "run this tool with the apply command."
        )
        return 0

    if args.command == "apply":
        stats = apply_template(config, args.splits, template_path)
        print(
            f"[OK] applied template={template_path} pages={stats['pages']} "
            f"words={stats['words']}"
        )
        for split, manifest in output_manifests(config, args.splits).items():
            print(f"[OK] {split}_manifest={manifest}")
        return 0

    stats = audit_e2e_manifests(config, args.splits)
    print(f"[OK] E2E audit passed: pages={stats['pages']} words={stats['words']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
