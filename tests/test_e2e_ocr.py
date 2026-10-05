import inspect
import json
import random
import runpy
from pathlib import Path

from PIL import Image

from manchu_ocr.metrics.e2e_ocr_metrics import (
    aggregate_edit_counts,
    build_e2e_character_pairs,
    compute_e2e_recognition_metrics,
    match_word_items,
    page_transcription_metrics,
    parse_page_annotation,
)
from manchu_ocr.pipelines.full_ocr_pipeline import FullOCRPipeline
from manchu_ocr.data.transforms.det_transforms import build_det_transform
from manchu_ocr.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rectangle(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_iou_075_matching_uses_detection_evaluator_rule():
    gt_items = [{"box": rectangle(0, 0, 10, 10), "text": "abc"}]
    exact_pred = [{"box": rectangle(0, 0, 10, 10), "text": "abc"}]
    below_threshold_pred = [
        {"box": rectangle(2, 0, 12, 10), "text": "abc"}
    ]

    exact = match_word_items(exact_pred, gt_items)
    below = match_word_items(below_threshold_pred, gt_items)

    assert exact["tp"] == 1
    assert exact["fp"] == 0
    assert exact["fn"] == 0
    assert below["tp"] == 0
    assert below["fp"] == 1
    assert below["fn"] == 1


def test_e2e_cer_penalizes_missed_words_and_false_positive_text():
    gt_items = [
        {"box": rectangle(0, 0, 10, 10), "text": "abc"},
        {"box": rectangle(20, 0, 30, 10), "text": "de"},
    ]
    pred_items = [
        {"box": rectangle(0, 0, 10, 10), "text": "axc"},
        {"box": rectangle(40, 0, 50, 10), "text": "q"},
    ]
    matching = match_word_items(pred_items, gt_items, iou_threshold=0.75)
    counts = aggregate_edit_counts(
        build_e2e_character_pairs(pred_items, gt_items, matching)
    )

    assert matching["tp"] == 1
    assert matching["fn"] == 1
    assert matching["fp"] == 1
    assert counts["substitutions"] == 1
    assert counts["deletions"] == 2
    assert counts["insertions"] == 1
    assert counts["reference_characters"] == 5
    assert counts["cer"] == 0.8


def test_e2e_metrics_use_strict_recognition_for_matched_crops():
    gt_items = [
        {"box": rectangle(0, 0, 10, 10), "text": "abc"},
        {"box": rectangle(20, 0, 30, 10), "text": "de"},
    ]
    pred_items = [
        {"box": rectangle(0, 0, 10, 10), "text": "axc"},
        {"box": rectangle(40, 0, 50, 10), "text": "q"},
    ]
    matching = match_word_items(pred_items, gt_items, iou_threshold=0.75)
    metrics = compute_e2e_recognition_metrics(
        pred_items,
        gt_items,
        matching,
    )

    assert metrics["strict_cer"] == 0.8
    assert metrics["cer"] == 0.8
    assert metrics["word_accuracy"] == 0.0
    assert metrics["strict_word_accuracy"] == 0.0


def test_detector_degradation_stage_preserves_transcription_and_gt_geometry():
    transform = build_det_transform(
        augment=True,
        augmentation_cfg={
            "nonuniform_scale_prob": 1.0,
            "horizontal_scale_min": 0.8,
            "horizontal_scale_max": 0.8,
            "vertical_scale_min": 1.0,
            "vertical_scale_max": 1.0,
            "elastic_prob": 0.0,
            "blur_prob": 0.0,
            "noise_contrast_prob": 0.0,
        },
    )
    image = Image.new("RGB", (100, 100), color="white")
    items = [{"box": rectangle(10, 20, 30, 40), "points": rectangle(10, 20, 30, 40), "text": "abc"}]
    random.seed(42)
    degraded, transformed, meta = transform.apply_augmentation(image, items)

    assert degraded.size == image.size
    assert meta["enabled"] is True
    assert transformed[0]["text"] == "abc"
    assert transformed[0]["points"] != items[0]["points"]


def test_page_reading_order_is_column_then_vertical():
    items = [
        {"index": 0, "box": rectangle(10, 80, 30, 100), "text": "b"},
        {"index": 1, "box": rectangle(100, 20, 120, 40), "text": "c"},
        {"index": 2, "box": rectangle(12, 10, 32, 30), "text": "a"},
    ]

    columns = FullOCRPipeline.group_columns(items)
    ordered = FullOCRPipeline.sort_items_reading_order(items)

    assert [item["text"] for item in ordered] == ["a", "b", "c"]
    assert FullOCRPipeline.build_page_text(columns) == "a b\nc"


def test_page_metrics_include_word_and_column_separators():
    metrics = page_transcription_metrics("ab c\nd", "ab x\nd")
    assert metrics["substitutions"] == 1
    assert metrics["deletions"] == 0
    assert metrics["insertions"] == 0
    assert metrics["reference_characters"] == len("ab x\nd")


def test_generic_detection_label_is_not_accepted_as_transcription(tmp_path):
    annotation_path = tmp_path / "page.json"
    annotation_path.write_text(
        json.dumps(
            {
                "polygons": [
                    {
                        "label": "text",
                        "points": rectangle(0, 0, 10, 20),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    items = parse_page_annotation(annotation_path)
    assert items[0]["text"] == ""


def test_full_pipeline_crops_detector_box_and_has_no_gt_input(tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), color="white").save(image_path)
    seen_crop_sizes = []

    class FakeDetector:
        def __call__(self, image):
            assert image.size == (100, 100)
            return {
                "boxes": [rectangle(10, 20, 30, 50)],
                "scores": [0.9],
            }

    class FakeRecognizer:
        def __call__(self, crop):
            seen_crop_sizes.append(crop.size)
            return {"text": "word", "raw_text": "word", "confidence": 0.8}

    pipeline = FullOCRPipeline.__new__(FullOCRPipeline)
    pipeline.detector = FakeDetector()
    pipeline.recognizer = FakeRecognizer()
    pipeline.crop_padding = 0
    pipeline.crop_padding_ratio = 0.0

    result = pipeline(image_path)

    assert list(inspect.signature(FullOCRPipeline.__call__).parameters) == [
        "self",
        "image_path",
    ]
    assert seen_crop_sizes == [(20, 30)]
    assert result["uses_ground_truth_crops"] is False
    assert result["items"][0]["detector_index"] == 0
    assert result["items"][0]["crop_source"] == "detector_prediction"


def test_e2e_plan_contains_exactly_the_requested_four_combinations():
    plan = load_yaml(PROJECT_ROOT / "configs" / "experiments" / "e2e_ocr.yaml")
    combinations = {
        (
            Path(item["detector_config"]).name,
            Path(item["recognizer_config"]).name,
        )
        for item in plan["pipelines"]
    }
    assert plan["protocol"]["strict_evaluation"] is True
    assert plan["protocol"]["detector_iou_threshold"] == 0.75
    assert plan["protocol"]["detector_degradation"] is False
    assert plan["protocol"]["recognition_metric_tolerance"] is False
    assert plan["protocol"]["uses_ground_truth_crops"] is False
    assert combinations == {
        ("dbnetpp_official_baseline.yaml", "svtr_official_baseline.yaml"),
        ("dbnetpp_vsaa_asym_shrink.yaml", "svtr_official_dab_lortho.yaml"),
        ("dbnetpp_official_baseline.yaml", "svtr_official_dab_lortho.yaml"),
        ("dbnetpp_vsaa_asym_shrink.yaml", "svtr_official_baseline.yaml"),
    }


def test_e2e_validation_and_test_commands_share_protocol():
    namespace = runpy.run_path(
        str(PROJECT_ROOT / "scripts" / "run_e2e_ocr_experiments.py")
    )
    Evaluation = namespace["E2EPipelineEvaluation"]
    build_command = namespace["build_evaluation_command"]
    plan = load_yaml(PROJECT_ROOT / "configs" / "experiments" / "e2e_ocr.yaml")
    evaluation = Evaluation(
        pipeline_id="dbnetpp_svtr",
        pipeline_name="DBNet++ + SVTR",
        detector_name="dbnetpp_official_baseline",
        recognizer_name="svtr_official_baseline",
        detector_config=Path("det.yaml"),
        recognizer_config=Path("rec.yaml"),
        detector_checkpoint=Path("det.pth"),
        recognizer_checkpoint=Path("rec.pth"),
        detector_checkpoint_tag="best",
        recognizer_checkpoint_tag="best",
        detector_iou_threshold=0.75,
    )
    common = {
        "evaluation": evaluation,
        "output_root": Path("outputs"),
        "plan": plan,
        "python_bin": "python",
        "device": "cuda",
        "save_records": True,
    }
    validation = build_command(
        manifest=Path("val.txt"), split="validation", **common
    )
    test = build_command(manifest=Path("test.txt"), split="test", **common)

    for option in ("--manifest", "--split", "--output-dir"):
        validation[validation.index(option) + 1] = f"<{option}>"
        test[test.index(option) + 1] = f"<{option}>"
    assert validation == test
    assert "--degradation-policy" not in validation
    assert "--iou-threshold" not in validation


def test_e2e_transcription_template_preserves_split_and_source_annotation(tmp_path):
    namespace = runpy.run_path(
        str(PROJECT_ROOT / "tools" / "prepare_e2e_transcriptions.py")
    )
    export_template = namespace["export_template"]
    read_template = namespace["read_template"]
    write_template = namespace["write_template"]
    apply_template = namespace["apply_template"]

    page_path = tmp_path / "page.png"
    source_annotation = tmp_path / "source.json"
    source_manifest = tmp_path / "val_source.txt"
    Image.new("RGB", (50, 80), color="white").save(page_path)
    source_annotation.write_text(
        json.dumps(
            {
                "polygons": [
                    {
                        "label": "text",
                        "points": rectangle(10, 10, 30, 60),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_manifest.write_text(
        f"{page_path}\t{source_annotation}\n",
        encoding="utf-8",
    )
    template_path = tmp_path / "e2e" / "transcriptions.csv"
    output_manifest = tmp_path / "e2e" / "val.txt"
    config = {
        "detection_data": {"val_list": str(source_manifest)},
        "e2e_data": {
            "annotation_dir": str(tmp_path / "e2e" / "annotations"),
            "review_crop_dir": str(tmp_path / "e2e" / "crops"),
            "transcription_template": str(template_path),
            "val_list": str(output_manifest),
        },
    }

    stats = export_template(
        config,
        ["val"],
        template_path,
        tmp_path / "e2e" / "crops",
        True,
        0,
    )
    assert stats["words"] == 1
    rows = read_template(template_path)
    rows[0]["transcription"] = "manju"
    rows[0]["status"] = "done"
    write_template(rows, template_path)
    apply_template(config, ["val"], template_path)

    source_data = json.loads(source_annotation.read_text(encoding="utf-8"))
    assert "transcription" not in source_data["polygons"][0]
    output_line = output_manifest.read_text(encoding="utf-8").strip()
    output_image, output_annotation = output_line.split("\t")
    assert Path(output_image) == page_path
    enriched = json.loads(Path(output_annotation).read_text(encoding="utf-8"))
    assert enriched["polygons"][0]["transcription"] == "manju"
