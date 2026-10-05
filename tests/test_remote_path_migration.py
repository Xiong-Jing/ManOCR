import csv
import io
import json
from pathlib import Path

import pytest

from manchu_ocr.utils.config import load_yaml
from tools.migrate_remote_paths import (
    FileChange,
    MigrationPlan,
    PathMapper,
    apply_plan,
    build_plan,
    migrate_csv,
    migrate_json,
    migrate_manifest,
    validate_manifest_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mapper():
    return PathMapper({
        "OCR_Manchu": "/root/code/OCR_Manchu",
        "Manchu_Detection_Data": "/root/code/Manchu_Detection_Data",
        "Manchu_Recognition_Data": "/root/code/Manchu_Recognition_Data",
    })


@pytest.mark.parametrize("old", [
    "C:/Users/ahs/Desktop/Manchu_Detection_Data/raw/images/满文  页.png",
    r"C:\Users\ahs\Desktop\Manchu_Detection_Data\raw\images\满文  页.png",
    "/old/server/Manchu_Detection_Data/raw/images/满文  页.png",
    "/root/code/Manchu_Detection_Data/raw/images/满文  页.png",
])
def test_mapper_preserves_filename_and_is_idempotent(mapper, old):
    expected = "/root/code/Manchu_Detection_Data/raw/images/满文  页.png"
    assert mapper(old) == expected
    assert mapper(expected) == expected
    assert mapper("file_Manchu_Detection_Data.png") == "file_Manchu_Detection_Data.png"


def test_mapper_rejects_path_traversal(mapper):
    with pytest.raises(ValueError, match="traversal"):
        mapper("C:/Manchu_Detection_Data/../outside.txt")


def test_recognition_manifest_preserves_bom_newlines_order_and_labels(mapper):
    original = (
        "\ufeffC:/old/Manchu_Recognition_Data/raw/word_images/1.jpg\t  ab a  \r\n"
        "\r\n"
        "C:/old/Manchu_Recognition_Data/raw/word_images/2.jpg\tOCR_Manchu\r\n"
    ).encode("utf-8")
    migrated, count, rows, targets = migrate_manifest(original, mapper, recognition=True)
    assert count == rows == 2
    assert migrated.startswith(b"\xef\xbb\xbf")
    assert migrated.decode("utf-8").splitlines()[0].endswith("\t  ab a  ")
    assert migrated.decode("utf-8").splitlines()[2].endswith("\tOCR_Manchu")
    assert migrated.count(b"\r\n") == 3
    assert len(targets) == 2


def test_detection_manifest_rewrites_both_paths(mapper):
    original = (
        "C:/old/Manchu_Detection_Data/raw/images/1.png\t"
        "C:/old/Manchu_Detection_Data/processed/annotations_clean/1.json\n"
    ).encode()
    migrated, count, rows, targets = migrate_manifest(original, mapper, recognition=False)
    assert count == 2 and rows == 1 and len(targets) == 2
    assert "C:/old" not in migrated.decode()
    assert migrated.count(b"\t") == 1


def test_csv_rewrites_only_path_columns(mapper):
    source = io.StringIO(newline="")
    writer = csv.DictWriter(source, fieldnames=[
        "image_path", "source_annotation_path", "review_crop_path",
        "transcription", "status", "notes",
    ])
    writer.writeheader()
    writer.writerow({
        "image_path": "C:/old/Manchu_Detection_Data/raw/images/1.png",
        "source_annotation_path": "C:/old/Manchu_Detection_Data/processed/annotations_clean/1.json",
        "review_crop_path": "C:/old/Manchu_Detection_Data/processed/e2e/review_crops/1.png",
        "transcription": "  a, b  ",
        "status": "done",
        "notes": "C:/old/Manchu_Detection_Data/this_is_not_a_path_field",
    })
    content = source.getvalue().encode("utf-8")
    migrated, count = migrate_csv(content, mapper)
    row = next(csv.DictReader(io.StringIO(migrated.decode("utf-8"))))
    assert count == 3
    assert row["transcription"] == "  a, b  "
    assert row["status"] == "done"
    assert row["notes"].startswith("C:/old/")


def test_json_does_not_modify_geometry_or_transcription(mapper):
    data = {
        "image_path": "C:/old/Manchu_Detection_Data/raw/images/1.png",
        "source_json": "C:/old/Manchu_Detection_Data/raw/annotations_json/1.json",
        "polygons": [{
            "points": [[1.0, 2.0], [10.0, 20.0]],
            "transcription": "Manchu_Detection_Data/a",
            "label": "text",
        }],
    }
    migrated, count = migrate_json(json.dumps(data).encode(), mapper)
    result = json.loads(migrated)
    assert count == 2
    assert result["polygons"] == data["polygons"]


def test_full_migration_has_backups_and_preserves_splits_and_auxiliary_files(tmp_path):
    det = tmp_path / "Manchu_Detection_Data"
    rec = tmp_path / "Manchu_Recognition_Data"
    project = tmp_path / "OCR_Manchu"
    for root in (det, rec, project):
        root.mkdir()
    image = det / "raw/images/满文  页.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    annotation = det / "processed/annotations_clean/page.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps({
        "image_path": "C:/old/Manchu_Detection_Data/raw/images/满文  页.png",
        "polygons": [{"points": [[1, 2], [3, 4]], "label": "text"}],
    }), encoding="utf-8")
    word = rec / "raw/word_images/word.jpg"
    word.parent.mkdir(parents=True)
    word.write_bytes(b"word")
    (rec / "processed").mkdir()
    config = {
        "project_root": str(project),
        "detection_data": {"root": str(det), "clean_annotations": str(annotation.parent)},
        "recognition_data": {"root": str(rec), "labels_csv": str(rec / "processed/labels.csv")},
    }
    originals = {}
    for split in ("train", "val", "test"):
        for section, root, content in (
            ("detection_data", det, (
                "C:/old/Manchu_Detection_Data/raw/images/满文  页.png\t"
                "C:/old/Manchu_Detection_Data/processed/annotations_clean/page.json\n"
            )),
            ("recognition_data", rec, "C:/old/Manchu_Recognition_Data/raw/word_images/word.jpg\tab a\n"),
        ):
            manifest = root / "processed" / (split + ".txt")
            manifest.write_text(content, encoding="utf-8")
            originals[manifest.resolve()] = manifest.read_bytes()
            config[section][split + "_list"] = str(manifest)
    labels = rec / "processed/labels.csv"
    labels.write_text(
        "image_path,label\nC:/old/Manchu_Recognition_Data/raw/word_images/word.jpg,ab a\n",
        encoding="utf-8",
    )
    charset = rec / "processed/charset.txt"
    transition = rec / "processed/transition_matrix.npy"
    charset.write_bytes(b"a\nb\n")
    transition.write_bytes(b"unchanged matrix")
    mapping = PathMapper({
        "OCR_Manchu": str(project),
        "Manchu_Detection_Data": str(det),
        "Manchu_Recognition_Data": str(rec),
    })
    plan = build_plan(config, mapping)
    assert list(plan.manifest_rows.values()) == [1] * 6
    validate_manifest_files(plan)
    original_bytes = {change.path: change.before for change in plan.changes}
    backups = apply_plan(plan)
    assert len(backups) == len(plan.changes) == 8
    for backup, change in zip(backups, plan.changes):
        assert backup.read_bytes() == original_bytes[change.path]
    for manifest in originals:
        assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1
        if manifest.is_relative_to(rec):
            assert manifest.read_text(encoding="utf-8").endswith("\tab a\n")
    assert charset.read_bytes() == b"a\nb\n"
    assert transition.read_bytes() == b"unchanged matrix"
    assert json.loads(annotation.read_text(encoding="utf-8"))["polygons"] == [
        {"points": [[1, 2], [3, 4]], "label": "text"},
    ]
    assert build_plan(config, mapping).changes == []


def test_missing_targets_stop_before_writes(tmp_path):
    manifest = tmp_path / "train.txt"
    manifest.write_bytes(b"original")
    plan = MigrationPlan(
        changes=[FileChange(manifest, b"original", b"changed", 1)],
        manifest_targets={str(tmp_path / "missing.jpg")},
        target_roots={str(tmp_path)},
    )
    with pytest.raises(FileNotFoundError, match="No file was modified"):
        validate_manifest_files(plan)
    assert manifest.read_bytes() == b"original"
    assert list(tmp_path.glob("*.paths_backup_*")) == []


def test_concurrent_changes_are_not_overwritten(tmp_path):
    path = tmp_path / "train.txt"
    path.write_bytes(b"user changed")
    plan = MigrationPlan(changes=[FileChange(path, b"original", b"new", 1)])
    with pytest.raises(RuntimeError, match="changed after planning"):
        apply_plan(plan)
    assert path.read_bytes() == b"user changed"
    assert list(tmp_path.glob("*.paths_backup_*")) == []


def test_all_model_configs_use_confirmed_server_roots(monkeypatch):
    for name in (
        "OCR_MANCHU_PROJECT_ROOT", "OCR_MANCHU_DET_ROOT",
        "OCR_MANCHU_REC_ROOT", "OCR_MANCHU_OUTPUT_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    config = load_yaml(PROJECT_ROOT / "configs/paths/remote_server.yaml")
    assert config["project_root"] == "/root/code/OCR_Manchu"
    assert config["detection_data"]["root"] == "/root/code/Manchu_Detection_Data"
    assert config["recognition_data"]["root"] == "/root/code/Manchu_Recognition_Data"
    assert config["outputs"]["root"] == "/root/code/OCR_Manchu/outputs"
    models = [
        *sorted((PROJECT_ROOT / "configs/detection").glob("*.yaml")),
        *sorted((PROJECT_ROOT / "configs/recognition").glob("*.yaml")),
    ]
    assert len(models) == 29
    for path in models:
        model = load_yaml(path)
        assert model["experiment"]["paths_config"] == "configs/paths/remote_server.yaml"
        if model.get("loss", {}).get("transition_matrix"):
            assert model["loss"]["transition_matrix"].startswith(
                "/root/code/Manchu_Recognition_Data/"
            )
