"""Migrate copied dataset paths without regenerating splits or annotations.

Dry-run is the default. Applying the plan validates all manifest targets first,
backs up every changed file, and updates only path fields in processed data.
Raw labels/images, charset, transition matrix, and historical results are not
modified.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from manchu_ocr.utils.config import load_yaml  # noqa: E402


@dataclass
class FileChange:
    path: Path
    before: bytes
    after: bytes
    replacements: int


@dataclass
class MigrationPlan:
    changes: list[FileChange] = field(default_factory=list)
    manifest_rows: dict[str, int] = field(default_factory=dict)
    manifest_targets: set[str] = field(default_factory=set)
    target_roots: set[str] = field(default_factory=set)
    scanned_files: int = 0


class PathMapper:
    def __init__(self, roots: Mapping[str, str]) -> None:
        self.roots = {
            name.casefold(): str(value).replace("\\", "/").rstrip("/")
            for name, value in roots.items()
        }

    def __call__(self, value: str) -> str:
        normalized = value.replace("\\", "/")
        parts = normalized.split("/")
        for index, component in enumerate(parts):
            root = self.roots.get(component.casefold())
            if root is not None:
                tail = parts[index + 1:]
                if ".." in tail:
                    raise ValueError(f"Refusing path traversal in {value!r}")
                return "/".join([root, *[part for part in tail if part not in ("", ".")]])
        # Do not guess a mapping for an unrelated path or filename.
        return value


def _decode(content: bytes) -> tuple[str, bool]:
    return content.decode("utf-8-sig"), content.startswith(b"\xef\xbb\xbf")


def _encode(text: str, bom: bool) -> bytes:
    return (b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8")


def migrate_manifest(
    content: bytes,
    mapper: PathMapper,
    *,
    recognition: bool,
) -> tuple[bytes, int, int, set[str]]:
    text, bom = _decode(content)
    output: list[str] = []
    replacements = rows = 0
    targets: set[str] = set()
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if not body.strip():
            output.append(line)
            continue
        fields = body.split("\t")
        if len(fields) != 2 or not fields[0].strip():
            raise ValueError(f"Invalid manifest row {line_number}: expected two columns")
        # Recognition labels remain byte-for-byte unchanged, including spaces.
        for index in range(1 if recognition else 2):
            if not fields[index].strip():
                raise ValueError(f"Empty manifest path at row {line_number}")
            mapped = mapper(fields[index])
            replacements += int(mapped != fields[index])
            fields[index] = mapped
            targets.add(mapped)
        output.append("\t".join(fields) + ending)
        rows += 1
    return _encode("".join(output), bom), replacements, rows, targets


def migrate_csv(content: bytes, mapper: PathMapper) -> tuple[bytes, int]:
    text, bom = _decode(content)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = reader.fieldnames
    if not fieldnames:
        return content, 0
    path_columns = [
        name for name in fieldnames
        if name in {"image_path", "source_annotation_path", "review_crop_path"}
    ]
    if not path_columns:
        return content, 0
    rows = []
    replacements = 0
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Invalid CSV row: column count does not match header")
        for column in path_columns:
            mapped = mapper(row[column])
            replacements += int(mapped != row[column])
            row[column] = mapped
        rows.append(row)
    if not replacements:
        return content, 0
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\r\n" if "\r\n" in text else "\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return _encode(output.getvalue(), bom), replacements


def _is_path_key(key: str) -> bool:
    return key in {
        "image", "annotation", "imagePath", "source_json", "path",
        "e2e_transcription_source", "charset", "transition_matrix",
        "labels_csv", "train", "val", "test",
    } or key.endswith(("_path", "_dir", "_root", "_list"))


def migrate_json(content: bytes, mapper: PathMapper) -> tuple[bytes, int]:
    text, bom = _decode(content)
    data = json.loads(text)
    replacements = 0

    def visit(value: Any) -> Any:
        nonlocal replacements
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if isinstance(item, str) and _is_path_key(key):
                    mapped = mapper(item)
                    replacements += int(mapped != item)
                    result[key] = mapped
                else:
                    result[key] = visit(item)
            return result
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    migrated = visit(data)
    if not replacements:
        return content, 0
    return _encode(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", bom), replacements


def build_plan(config: Mapping[str, Any], mapper: PathMapper) -> MigrationPlan:
    plan = MigrationPlan()
    plan.target_roots.update(mapper.roots.values())
    seen: set[Path] = set()
    source_roots = [
        Path(config["project_root"]).resolve(),
        Path(config["detection_data"]["root"]).resolve(),
        Path(config["recognition_data"]["root"]).resolve(),
    ]

    def add(path: Path, kind: str, *, recognition: bool = False, required: bool = False) -> None:
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"Missing prepared manifest: {path}")
            return
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in source_roots):
            raise ValueError(f"Refusing to modify file outside configured roots: {path}")
        if resolved in seen:
            return
        seen.add(resolved)
        before = path.read_bytes()
        plan.scanned_files += 1
        if kind == "manifest":
            after, count, rows, targets = migrate_manifest(
                before, mapper, recognition=recognition,
            )
            if required and not rows:
                raise ValueError(f"Prepared manifest is empty: {path}")
            plan.manifest_rows[str(path)] = rows
            plan.manifest_targets.update(targets)
        elif kind == "csv":
            after, count = migrate_csv(before, mapper)
        else:
            after, count = migrate_json(before, mapper)
        if count:
            plan.changes.append(FileChange(resolved, before, after, count))

    for section_name in ("detection_data", "recognition_data", "e2e_data"):
        section = config.get(section_name, {})
        for split_key in ("train_list", "val_list", "test_list"):
            if section.get(split_key):
                add(
                    Path(section[split_key]), "manifest",
                    recognition=section_name == "recognition_data",
                    required=section_name != "e2e_data",
                )

    det = config["detection_data"]
    rec = config["recognition_data"]
    e2e = config.get("e2e_data", {})
    json_dirs = [Path(det["clean_annotations"])]
    if e2e.get("annotation_dir"):
        json_dirs.append(Path(e2e["annotation_dir"]))
    for directory in json_dirs:
        for path in sorted(directory.rglob("*.json")):
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"Annotation symlink escapes configured directory: {path}")
            add(path, "json")
    for section in (det, rec):
        processed = Path(section["root"]) / "processed"
        for path in sorted(processed.glob("*.json")):
            add(path, "json")
    add(Path(rec["labels_csv"]), "csv")
    add(Path(rec["labels_csv"]).with_name("labels_normalized.csv"), "csv")
    if e2e.get("transcription_template"):
        add(Path(e2e["transcription_template"]), "csv")
    legacy_manifests = Path(config["project_root"]) / "data" / "manifests"
    for pattern, recognition in (
        ("detection_*.txt", False), ("full_ocr_*.txt", False),
        ("recognition_*.txt", True),
    ):
        for path in sorted(legacy_manifests.glob(pattern)):
            add(path, "manifest", recognition=recognition)
    return plan


def validate_manifest_files(plan: MigrationPlan) -> None:
    roots = [Path(root).resolve() for root in plan.target_roots]
    unrelated = [
        path for path in sorted(plan.manifest_targets)
        if not Path(path).is_absolute()
        or not any(Path(path).resolve().is_relative_to(root) for root in roots)
    ]
    if unrelated:
        raise ValueError(
            "Unmapped manifest path(s) outside deployment roots: "
            + ", ".join(unrelated[:10])
        )
    missing = [path for path in sorted(plan.manifest_targets) if not Path(path).is_file()]
    if missing:
        examples = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} mapped manifest target(s) are missing:\n{examples}\n"
            "No file was modified. Check upload layout before applying."
        )


def apply_plan(plan: MigrationPlan) -> list[Path]:
    # Reject concurrent changes before making even the backups.
    for change in plan.changes:
        if change.path.read_bytes() != change.before:
            raise RuntimeError(f"File changed after planning: {change.path}")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    backups = []
    for change in plan.changes:
        backup = change.path.with_name(change.path.name + ".paths_backup_" + tag)
        shutil.copy2(change.path, backup)
        backups.append(backup)
    written: list[tuple[FileChange, Path]] = []
    try:
        for change, backup in zip(plan.changes, backups):
            temporary = change.path.with_name("." + change.path.name + "." + tag + ".tmp")
            try:
                temporary.write_bytes(change.after)
                shutil.copymode(change.path, temporary)
                os.replace(temporary, change.path)
                written.append((change, backup))
            finally:
                temporary.unlink(missing_ok=True)
    except Exception:
        for change, backup in reversed(written):
            shutil.copy2(backup, change.path)
        raise
    return backups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paths/remote_server.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only (default).")
    mode.add_argument("--apply", action="store_true", help="Back up and apply path replacements.")
    parser.add_argument(
        "--skip-file-check", action="store_true",
        help="Preview mappings without checking target files; not allowed with --apply.",
    )
    args = parser.parse_args(argv)
    if args.apply and args.skip_file_check:
        parser.error("--skip-file-check is only allowed for a dry-run")
    try:
        config = load_yaml(args.config)
        mapper = PathMapper({
            "OCR_Manchu": config["project_root"],
            "Manchu_Detection_Data": config["detection_data"]["root"],
            "Manchu_Recognition_Data": config["recognition_data"]["root"],
        })
        plan = build_plan(config, mapper)
        for path, rows in plan.manifest_rows.items():
            print(f"[MANIFEST] rows={rows} {path}", flush=True)
        print(
            f"[PLAN] scanned={plan.scanned_files} changed={len(plan.changes)} "
            f"path_replacements={sum(change.replacements for change in plan.changes)}",
            flush=True,
        )
        print("[PLAN] split membership/order, labels, boxes and loss/metric configs are unchanged.")
        if not args.skip_file_check:
            print(f"[CHECK] checking {len(plan.manifest_targets)} unique manifest targets...", flush=True)
            validate_manifest_files(plan)
        if args.apply:
            backups = apply_plan(plan)
            print(f"[OK] Updated {len(backups)} file(s); original backups use .paths_backup_<UTC> suffix.")
        else:
            print("[OK] Dry-run only. No files were modified; use --apply after confirming paths.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
