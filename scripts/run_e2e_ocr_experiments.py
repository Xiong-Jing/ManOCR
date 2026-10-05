from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manchu_ocr.metrics.e2e_ocr_metrics import parse_page_annotation  # noqa: E402
from manchu_ocr.utils.config import load_yaml  # noqa: E402
from manchu_ocr.utils.experiment_result import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    normalize_result_path,
    normalize_split_name,
    split_key,
    utc_now,
)
from manchu_ocr.utils.file_io import save_json  # noqa: E402


DEFAULT_PLAN = PROJECT_ROOT / "configs" / "experiments" / "e2e_ocr.yaml"
STRICT_IOU_THRESHOLD = 0.75


@dataclass(frozen=True)
class E2EPipelineEvaluation:
    pipeline_id: str
    pipeline_name: str
    detector_name: str
    recognizer_name: str
    detector_config: Path
    recognizer_config: Path
    detector_checkpoint: Path
    recognizer_checkpoint: Path
    detector_checkpoint_tag: str
    recognizer_checkpoint_tag: str
    detector_iou_threshold: float


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def resolve_output_root(plan: Mapping[str, Any], override: str | None) -> Path:
    if override:
        output_root = Path(override)
    elif os.environ.get("OCR_MANCHU_OUTPUT_ROOT"):
        output_root = Path(os.environ["OCR_MANCHU_OUTPUT_ROOT"])
    else:
        paths_cfg = load_yaml(_project_path(plan["data"]["paths_config"]))
        output_root = Path(paths_cfg["outputs"]["root"])
    return output_root if output_root.is_absolute() else PROJECT_ROOT / output_root


def resolve_manifests(plan: Mapping[str, Any]) -> dict[str, Path]:
    data_plan = plan["data"]
    paths_cfg = load_yaml(_project_path(data_plan["paths_config"]))
    data_section = paths_cfg[data_plan["page_manifest_section"]]
    keys = data_plan["split_manifest_keys"]
    return {
        normalize_split_name(split): Path(data_section[manifest_key])
        for split, manifest_key in keys.items()
    }


def select_checkpoint(
    output_root: Path,
    task: str,
    experiment_name: str,
    policy: str,
) -> tuple[Path | None, str | None]:
    checkpoint_dir = output_root / "checkpoints" / task / experiment_name
    best_checkpoint = checkpoint_dir / "best.pth"
    last_checkpoint = checkpoint_dir / "last.pth"
    if best_checkpoint.is_file():
        return best_checkpoint, "best"
    if policy == "best_then_last" and last_checkpoint.is_file():
        return last_checkpoint, "last"
    return None, None


def build_pipeline_evaluations(
    plan: Mapping[str, Any],
    output_root: Path,
    checkpoint_policy: str,
    pipeline_filter: set[str] | None = None,
) -> tuple[list[E2EPipelineEvaluation], list[str]]:
    evaluations: list[E2EPipelineEvaluation] = []
    issues: list[str] = []

    for pipeline in plan["pipelines"]:
        pipeline_id = str(pipeline["id"])
        if pipeline_filter and pipeline_id not in pipeline_filter:
            continue

        detector_config = _project_path(pipeline["detector_config"])
        recognizer_config = _project_path(pipeline["recognizer_config"])
        for role, config_path in (
            ("detector", detector_config),
            ("recognizer", recognizer_config),
        ):
            if not config_path.is_file():
                issues.append(
                    f"{pipeline_id}: missing {role} config: {config_path}"
                )
        if not detector_config.is_file() or not recognizer_config.is_file():
            continue

        detector_cfg = load_yaml(detector_config)
        recognizer_cfg = load_yaml(recognizer_config)
        detector_name = str(detector_cfg["experiment"]["name"])
        recognizer_name = str(recognizer_cfg["experiment"]["name"])
        configured_iou = float(
            detector_cfg.get("eval", {}).get("iou_thresh", STRICT_IOU_THRESHOLD)
        )
        if abs(configured_iou - STRICT_IOU_THRESHOLD) > 1e-12:
            issues.append(
                f"{pipeline_id}: detector IoU mismatch for {detector_name}: "
                f"config={configured_iou}, expected={STRICT_IOU_THRESHOLD}"
            )
        detector_checkpoint, detector_tag = select_checkpoint(
            output_root,
            "detection",
            detector_name,
            checkpoint_policy,
        )
        recognizer_checkpoint, recognizer_tag = select_checkpoint(
            output_root,
            "recognition",
            recognizer_name,
            checkpoint_policy,
        )
        if detector_checkpoint is None or detector_tag is None:
            issues.append(
                f"{pipeline_id}: missing detector checkpoint: "
                f"{output_root / 'checkpoints' / 'detection' / detector_name}"
            )
        if recognizer_checkpoint is None or recognizer_tag is None:
            issues.append(
                f"{pipeline_id}: missing recognizer checkpoint: "
                f"{output_root / 'checkpoints' / 'recognition' / recognizer_name}"
            )
        if (
            detector_checkpoint is None
            or detector_tag is None
            or recognizer_checkpoint is None
            or recognizer_tag is None
        ):
            continue

        evaluations.append(
            E2EPipelineEvaluation(
                pipeline_id=pipeline_id,
                pipeline_name=str(pipeline["name"]),
                detector_name=detector_name,
                recognizer_name=recognizer_name,
                detector_config=detector_config,
                recognizer_config=recognizer_config,
                detector_checkpoint=detector_checkpoint,
                recognizer_checkpoint=recognizer_checkpoint,
                detector_checkpoint_tag=detector_tag,
                recognizer_checkpoint_tag=recognizer_tag,
                detector_iou_threshold=STRICT_IOU_THRESHOLD,
            )
        )

    return evaluations, list(dict.fromkeys(issues))


def preflight_manifest(manifest_path: Path) -> dict[str, int]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing E2E page manifest: {manifest_path}")

    page_count = 0
    word_count = 0
    missing_text: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid E2E manifest line {line_index}: {line!r}"
                )
            image_path, annotation_path = map(Path, parts)
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing page image: {image_path}")
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Missing page annotation: {annotation_path}"
                )
            items = parse_page_annotation(annotation_path)
            page_count += 1
            word_count += len(items)
            missing_text.extend(
                f"{annotation_path}#item[{item['source_index']}]"
                for item in items
                if not item["text"]
            )

    if page_count == 0:
        raise RuntimeError(f"No pages found in manifest: {manifest_path}")
    if missing_text:
        examples = "\n".join(f"  - {item}" for item in missing_text[:10])
        raise RuntimeError(
            f"{manifest_path} contains {len(missing_text)} GT word(s) without "
            "real transcription. Generic labels such as 'text' cannot support "
            "E2E WA/CER. Add text/transcription fields without changing the page "
            f"split. Examples:\n{examples}"
        )
    return {"pages": page_count, "words": word_count}


def build_evaluation_command(
    evaluation: E2EPipelineEvaluation,
    manifest: Path,
    split: str,
    output_root: Path,
    plan: Mapping[str, Any],
    python_bin: str,
    device: str,
    save_records: bool,
) -> list[str]:
    output_dir = (
        output_root
        / str(plan["outputs"]["metrics_subdir"])
        / evaluation.pipeline_id
    )
    runtime = plan["target_runtime"]
    command = [
        python_bin,
        str(_project_path(plan["workflow"]["evaluator"])),
        "--det-config",
        str(evaluation.detector_config),
        "--det-checkpoint",
        str(evaluation.detector_checkpoint),
        "--rec-config",
        str(evaluation.recognizer_config),
        "--rec-checkpoint",
        str(evaluation.recognizer_checkpoint),
        "--pipeline-name",
        evaluation.pipeline_name,
        "--manifest",
        str(manifest),
        "--split",
        split_key(split),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--crop-padding",
        str(runtime["crop_padding"]),
        "--crop-padding-ratio",
        str(runtime["crop_padding_ratio"]),
    ]
    if save_records:
        command.append("--save-records")
    return command


def result_path_for(
    output_root: Path,
    plan: Mapping[str, Any],
    evaluation: E2EPipelineEvaluation,
    split: str,
) -> Path:
    return (
        output_root
        / str(plan["outputs"]["metrics_subdir"])
        / evaluation.pipeline_id
        / f"eval_{split_key(split)}.json"
    )


def validate_result(
    result_path: Path,
    evaluation: E2EPipelineEvaluation,
    manifest: Path,
    split: str,
) -> dict[str, Any]:
    if not result_path.is_file():
        raise FileNotFoundError(f"Evaluator did not create: {result_path}")
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unexpected result schema in {result_path}")
    if result.get("task") != "e2e_ocr":
        raise ValueError(f"Unexpected task in {result_path}: {result.get('task')}")
    if result.get("model_name") != evaluation.pipeline_name:
        raise ValueError(f"Pipeline name mismatch in {result_path}")
    if result.get("split") != normalize_split_name(split):
        raise ValueError(f"Split mismatch in {result_path}")
    if normalize_result_path(result.get("manifest_path", "")) != normalize_result_path(
        manifest
    ):
        raise ValueError(f"Manifest mismatch in {result_path}")
    checkpoints = result.get("checkpoints", {})
    if normalize_result_path(checkpoints.get("detection", "")) != normalize_result_path(
        evaluation.detector_checkpoint
    ):
        raise ValueError(f"Detector checkpoint mismatch in {result_path}")
    if normalize_result_path(checkpoints.get("recognition", "")) != normalize_result_path(
        evaluation.recognizer_checkpoint
    ):
        raise ValueError(f"Recognizer checkpoint mismatch in {result_path}")

    metrics = result.get("metrics", {})
    required_metrics = {
        "e2e_word_accuracy",
        "e2e_exact_word_accuracy",
        "e2e_cer",
        "e2e_strict_cer",
        "page_transcription_cer",
        "page_transcription_score",
        "missed_words",
        "missed_word_rate",
    }
    missing = sorted(required_metrics.difference(metrics))
    if missing:
        raise ValueError(f"Missing E2E metrics {missing} in {result_path}")
    if metrics.get("uses_ground_truth_crops") is not False:
        raise ValueError(f"GT-crop provenance violation in {result_path}")
    if abs(
        float(metrics.get("iou_threshold", -1.0))
        - evaluation.detector_iou_threshold
    ) > 1e-12:
        raise ValueError(f"Strict detector IoU mismatch in {result_path}")
    if bool(metrics.get("degraded_evaluation")):
        raise ValueError(f"Evaluation degradation must be disabled in {result_path}")
    if bool(metrics.get("recognition_tolerance_applied", True)):
        raise ValueError(f"Recognition tolerance must be disabled in {result_path}")
    return result


COMPARISON_FIELDS = [
    "pipeline",
    "split",
    "detector",
    "recognizer",
    "detector_checkpoint",
    "recognizer_checkpoint",
    "manifest",
    "detector_iou_threshold",
    "strict_protocol",
    "e2e_word_accuracy",
    "e2e_exact_word_accuracy",
    "e2e_cer",
    "e2e_strict_cer",
    "page_transcription_cer",
    "page_transcription_score",
    "missed_words",
    "missed_word_rate",
    "false_positive_boxes",
    "detection_fmeasure",
    "runtime_seconds",
    "result_path",
]


def comparison_row(
    result: Mapping[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "pipeline": result["model_name"],
        "split": result["split"],
        "detector": result["models"]["detection"],
        "recognizer": result["models"]["recognition"],
        "detector_checkpoint": result["checkpoints"]["detection"],
        "recognizer_checkpoint": result["checkpoints"]["recognition"],
        "manifest": result["manifest_path"],
        "detector_iou_threshold": metrics["iou_threshold"],
        "strict_protocol": True,
        "e2e_word_accuracy": metrics["e2e_word_accuracy"],
        "e2e_exact_word_accuracy": metrics["e2e_exact_word_accuracy"],
        "e2e_cer": metrics["e2e_cer"],
        "e2e_strict_cer": metrics["e2e_strict_cer"],
        "page_transcription_cer": metrics["page_transcription_cer"],
        "page_transcription_score": metrics["page_transcription_score"],
        "missed_words": metrics["missed_words"],
        "missed_word_rate": metrics["missed_word_rate"],
        "false_positive_boxes": metrics["false_positive_boxes"],
        "detection_fmeasure": metrics["detection_fmeasure"],
        "runtime_seconds": result["runtime_seconds"],
        "result_path": normalize_result_path(result_path),
    }


def write_comparison(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    timestamp: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"e2e_comparison_{timestamp}.json"
    csv_path = output_dir / f"e2e_comparison_{timestamp}.csv"
    latest_json = output_dir / "e2e_comparison_latest.json"
    latest_csv = output_dir / "e2e_comparison_latest.csv"
    payload = {
        "schema_version": "ocr_manchu.e2e_comparison.v1",
        "generated_at_utc": utc_now(),
        "runs": list(rows),
    }
    save_json(payload, json_path)
    save_json(payload, latest_json)
    for path in (csv_path, latest_csv):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four formal detector x recognizer E2E OCR pipelines."
    )
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["validation", "test"],
        default=None,
    )
    parser.add_argument("--pipelines", nargs="*", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--checkpoint-policy",
        choices=["best_only", "best_then_last"],
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-save-records", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = load_yaml(_project_path(args.plan))
    selected_splits = [
        normalize_split_name(split)
        for split in (args.splits or plan["protocol"]["default_splits"])
    ]
    selected_pipeline_ids = None if not args.pipelines else set(args.pipelines)
    checkpoint_policy = str(
        args.checkpoint_policy or plan["protocol"]["checkpoint_policy"]
    )
    device = str(args.device or plan["target_runtime"]["device"])
    output_root = resolve_output_root(plan, args.output_root)
    manifests = resolve_manifests(plan)

    known_pipeline_ids = {str(item["id"]) for item in plan["pipelines"]}
    if selected_pipeline_ids:
        unknown = sorted(selected_pipeline_ids.difference(known_pipeline_ids))
        if unknown:
            print(f"[ERROR] Unknown pipeline id(s): {', '.join(unknown)}", file=sys.stderr)
            return 2

    evaluations, issues = build_pipeline_evaluations(
        plan,
        output_root,
        checkpoint_policy,
        selected_pipeline_ids,
    )
    manifest_stats: dict[str, dict[str, int]] = {}
    for split in selected_splits:
        try:
            manifest_stats[split] = preflight_manifest(manifests[split])
        except Exception as exc:
            issues.append(f"{split} data preflight failed: {exc}")

    expected_count = (
        len(selected_pipeline_ids)
        if selected_pipeline_ids is not None
        else len(plan["pipelines"])
    )
    print(f"[PLAN] target={plan['target_runtime']['hardware']}")
    print(f"[PLAN] device={device}")
    print(f"[PLAN] splits={','.join(selected_splits)}")
    print("[PLAN] detector protocol=original images, fixed IoU=0.75")
    print("[PLAN] recognition protocol=strict exact WA, standard CA/CER")
    print("[PLAN] crop_source=detector_predictions (GT crops disabled)")
    print(f"[PLAN] output_root={output_root}")
    print(f"[PLAN] pipelines_ready={len(evaluations)}/{expected_count}")
    for evaluation in evaluations:
        print(
            f"[PLAN] protocol={evaluation.pipeline_id} "
            f"IoU={evaluation.detector_iou_threshold} strict=true"
        )
    for split, stats in manifest_stats.items():
        print(
            f"[PLAN] {split}_manifest={manifests[split]} "
            f"pages={stats['pages']} words={stats['words']}"
        )

    if issues:
        for issue in list(dict.fromkeys(issues)):
            print(f"[PREFLIGHT] {issue}", file=sys.stderr)
        print(
            "[ERROR] Strict E2E preflight failed; no model inference was started.",
            file=sys.stderr,
        )
        return 2
    if len(evaluations) != expected_count:
        print("[ERROR] Not all requested pipelines are runnable.", file=sys.stderr)
        return 2

    save_records = bool(plan["outputs"].get("save_page_records", True))
    save_records = save_records and not args.no_save_records
    commands = [
        (
            evaluation,
            split,
            build_evaluation_command(
                evaluation,
                manifests[split],
                split,
                output_root,
                plan,
                args.python_bin,
                device,
                save_records,
            ),
        )
        for evaluation in evaluations
        for split in selected_splits
    ]
    for evaluation, split, command in commands:
        print(
            f"[COMMAND] pipeline={evaluation.pipeline_name} split={split} "
            f"det_checkpoint={evaluation.detector_checkpoint} "
            f"rec_checkpoint={evaluation.recognizer_checkpoint} "
            f"manifest={manifests[split]}"
        )
        print(f"[COMMAND] {shlex.join(command)}")
    if args.dry_run:
        print(f"[OK] Dry run validated {len(commands)} E2E run(s).")
        return 0

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SRC_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    environment.setdefault("OCR_MANCHU_PROJECT_ROOT", str(PROJECT_ROOT))
    environment["OCR_MANCHU_OUTPUT_ROOT"] = str(output_root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    comparison_dir = output_root / str(plan["outputs"]["metrics_subdir"])
    run_summary_path = comparison_dir / f"e2e_run_{timestamp}.json"
    summary: dict[str, Any] = {
        "schema_version": "ocr_manchu.e2e_run.v1",
        "experiment_name": plan["experiment"]["name"],
        "target_hardware": plan["target_runtime"]["hardware"],
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "splits": selected_splits,
        "evaluation_protocols": {
            evaluation.pipeline_id: {
                "iou_threshold": evaluation.detector_iou_threshold,
                "degradation_enabled": False,
                "recognition_metrics": "strict",
            }
            for evaluation in evaluations
        },
        "uses_ground_truth_crops": False,
        "planned_runs": len(commands),
        "runs": [],
    }
    comparison_rows: list[dict[str, Any]] = []
    failures = 0

    for evaluation, split, command in commands:
        started = perf_counter()
        print(f"[RUN] pipeline={evaluation.pipeline_name} split={split}")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        process_runtime = round(max(0.0, perf_counter() - started), 6)
        result_path = result_path_for(output_root, plan, evaluation, split)
        run_record: dict[str, Any] = {
            "pipeline_id": evaluation.pipeline_id,
            "pipeline_name": evaluation.pipeline_name,
            "split": split,
            "detector_checkpoint": normalize_result_path(
                evaluation.detector_checkpoint
            ),
            "recognizer_checkpoint": normalize_result_path(
                evaluation.recognizer_checkpoint
            ),
            "manifest": normalize_result_path(manifests[split]),
            "result_path": normalize_result_path(result_path),
            "process_runtime_seconds": process_runtime,
            "return_code": int(completed.returncode),
            "status": "completed" if completed.returncode == 0 else "failed",
        }
        if completed.returncode == 0:
            try:
                result = validate_result(
                    result_path,
                    evaluation,
                    manifests[split],
                    split,
                )
                run_record["metrics"] = result["metrics"]
                run_record["runtime_seconds"] = result["runtime_seconds"]
                comparison_rows.append(comparison_row(result, result_path))
            except Exception as exc:
                run_record["status"] = "invalid_result"
                run_record["error"] = str(exc)
                failures += 1
        else:
            failures += 1

        summary["runs"].append(run_record)
        summary["finished_at_utc"] = utc_now()
        save_json(summary, run_summary_path)
        write_comparison(comparison_rows, comparison_dir, timestamp)
        if run_record["status"] != "completed" and args.fail_fast:
            break

    summary["finished_at_utc"] = utc_now()
    summary["completed_runs"] = sum(
        int(item["status"] == "completed") for item in summary["runs"]
    )
    summary["failed_runs"] = failures
    save_json(summary, run_summary_path)
    comparison_json, comparison_csv = write_comparison(
        comparison_rows,
        comparison_dir,
        timestamp,
    )
    print(f"[RESULT] completed={summary['completed_runs']} failed={failures}")
    print(f"[RESULT] run_summary={run_summary_path}")
    print(f"[RESULT] comparison_json={comparison_json}")
    print(f"[RESULT] comparison_csv={comparison_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
