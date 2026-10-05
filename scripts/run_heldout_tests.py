from __future__ import annotations

import argparse
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

from manchu_ocr.utils.config import load_yaml  # noqa: E402
from manchu_ocr.utils.experiment_result import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    normalize_result_path,
    normalize_split_name,
    split_key,
    utc_now,
)
from manchu_ocr.utils.file_io import save_json  # noqa: E402


DEFAULT_PLAN = PROJECT_ROOT / "configs" / "experiments" / "heldout_all_models.yaml"
SUPPORTED_TASKS = ("recognition", "detection")


@dataclass(frozen=True)
class ModelEvaluation:
    task: str
    experiment_name: str
    model_architecture: str
    config_path: Path
    evaluator_path: Path
    checkpoint_path: Path
    checkpoint_tag: str
    manifests: Mapping[str, Path]
    batch_size: int
    num_workers: int
    save_predictions: bool
    evaluator_args: tuple[str, ...]


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def configured_model_paths(plan: Mapping[str, Any], task: str) -> list[Path]:
    task_cfg = plan["tasks"][task]
    return [_project_path(path) for path in task_cfg.get("configs", [])]


def resolve_output_root(
    plan: Mapping[str, Any],
    selected_tasks: Sequence[str],
    override: str | None,
) -> Path:
    if override:
        output_root = Path(override)
    elif os.environ.get("OCR_MANCHU_OUTPUT_ROOT"):
        output_root = Path(os.environ["OCR_MANCHU_OUTPUT_ROOT"])
    else:
        first_config = configured_model_paths(plan, selected_tasks[0])[0]
        model_cfg = load_yaml(first_config)
        paths_cfg = load_yaml(_project_path(model_cfg["experiment"]["paths_config"]))
        output_root = Path(paths_cfg["outputs"]["root"])

    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return output_root


def select_checkpoint(
    *,
    output_root: Path,
    checkpoint_subdir: str,
    experiment_name: str,
    policy: str,
) -> tuple[Path | None, str | None]:
    checkpoint_dir = output_root / "checkpoints" / checkpoint_subdir / experiment_name
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"

    if best_path.is_file():
        return best_path, "best"
    if policy == "best_then_last" and last_path.is_file():
        return last_path, "last"
    return None, None


def build_model_evaluations(
    *,
    plan: Mapping[str, Any],
    selected_tasks: Sequence[str],
    selected_splits: Sequence[str],
    output_root: Path,
    checkpoint_policy: str,
    model_filter: set[str] | None = None,
    batch_overrides: Mapping[str, int | None] | None = None,
    num_workers_override: int | None = None,
) -> tuple[list[ModelEvaluation], list[str]]:
    evaluations: list[ModelEvaluation] = []
    issues: list[str] = []
    batch_overrides = batch_overrides or {}

    for task in selected_tasks:
        task_cfg = plan["tasks"][task]
        evaluator_path = _project_path(task_cfg["evaluator"])
        if not evaluator_path.is_file():
            issues.append(f"missing evaluator: {evaluator_path}")
            continue

        for config_path in configured_model_paths(plan, task):
            if not config_path.is_file():
                issues.append(f"missing model config: {config_path}")
                continue

            model_cfg = load_yaml(config_path)
            experiment_name = str(model_cfg["experiment"]["name"])
            if model_filter and experiment_name not in model_filter:
                continue

            checkpoint_path, checkpoint_tag = select_checkpoint(
                output_root=output_root,
                checkpoint_subdir=str(task_cfg["checkpoint_subdir"]),
                experiment_name=experiment_name,
                policy=checkpoint_policy,
            )
            if checkpoint_path is None or checkpoint_tag is None:
                issues.append(
                    f"missing checkpoint ({checkpoint_policy}): "
                    f"{output_root / 'checkpoints' / str(task_cfg['checkpoint_subdir']) / experiment_name}"
                )
                continue

            paths_config_path = _project_path(model_cfg["experiment"]["paths_config"])
            paths_cfg = load_yaml(paths_config_path)
            data_cfg = paths_cfg[str(task_cfg["data_section"])]
            manifests = {
                "validation": Path(data_cfg["val_list"]),
                "test": Path(data_cfg["test_list"]),
            }
            for split_name in selected_splits:
                manifest_path = manifests[normalize_split_name(split_name)]
                if not manifest_path.is_file():
                    issues.append(
                        f"missing {task} {normalize_split_name(split_name)} manifest: "
                        f"{manifest_path}"
                    )

            batch_size = batch_overrides.get(task)
            if batch_size is None:
                batch_size = int(task_cfg["batch_size"])
            num_workers = (
                int(num_workers_override)
                if num_workers_override is not None
                else int(task_cfg["num_workers"])
            )

            evaluations.append(
                ModelEvaluation(
                    task=task,
                    experiment_name=experiment_name,
                    model_architecture=str(model_cfg["model"]["name"]),
                    config_path=config_path,
                    evaluator_path=evaluator_path,
                    checkpoint_path=checkpoint_path,
                    checkpoint_tag=checkpoint_tag,
                    manifests=manifests,
                    batch_size=int(batch_size),
                    num_workers=num_workers,
                    save_predictions=bool(task_cfg.get("save_predictions", True)),
                    evaluator_args=(),
                )
            )

    return evaluations, issues


def build_evaluation_command(
    evaluation: ModelEvaluation,
    *,
    split: str,
    python_bin: str,
) -> list[str]:
    """Build one command; validation and test differ only in --split."""
    command = [
        python_bin,
        str(evaluation.evaluator_path),
        "--config",
        str(evaluation.config_path),
        "--checkpoint",
        str(evaluation.checkpoint_path),
        "--split",
        split_key(split),
        "--batch-size",
        str(evaluation.batch_size),
        "--num-workers",
        str(evaluation.num_workers),
    ]
    command.extend(evaluation.evaluator_args)
    if evaluation.save_predictions:
        command.append("--save-predictions")
    return command


def result_path_for(
    output_root: Path,
    evaluation: ModelEvaluation,
    split: str,
) -> Path:
    return (
        output_root
        / "metrics"
        / evaluation.task
        / evaluation.experiment_name
        / f"eval_{split_key(split)}.json"
    )


def validate_result_file(
    result_path: Path,
    evaluation: ModelEvaluation,
    split: str,
) -> dict[str, Any]:
    if not result_path.is_file():
        raise FileNotFoundError(f"evaluator did not create result: {result_path}")

    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    required = {
        "schema_version",
        "model_name",
        "split",
        "checkpoint_path",
        "manifest_path",
        "metrics",
        "runtime_seconds",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"result is missing required fields {missing}: {result_path}")
    if result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unexpected result schema: {result['schema_version']}")
    if result["model_name"] != evaluation.experiment_name:
        raise ValueError(
            f"model mismatch: expected {evaluation.experiment_name}, "
            f"got {result['model_name']}"
        )
    if result["split"] != normalize_split_name(split):
        raise ValueError(
            f"split mismatch: expected {normalize_split_name(split)}, "
            f"got {result['split']}"
        )
    expected_checkpoint = normalize_result_path(evaluation.checkpoint_path)
    if normalize_result_path(result["checkpoint_path"]) != expected_checkpoint:
        raise ValueError(
            f"checkpoint mismatch: expected {expected_checkpoint}, "
            f"got {result['checkpoint_path']}"
        )
    expected_manifest = normalize_result_path(
        evaluation.manifests[normalize_split_name(split)]
    )
    if normalize_result_path(result["manifest_path"]) != expected_manifest:
        raise ValueError(
            f"manifest mismatch: expected {expected_manifest}, "
            f"got {result['manifest_path']}"
        )
    if not isinstance(result["metrics"], dict):
        raise TypeError("result metrics must be a mapping")
    if float(result["runtime_seconds"]) < 0:
        raise ValueError("runtime_seconds must be non-negative")

    return result


def _write_run_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(dict(summary), path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation and/or held-out test evaluation for every configured "
            "recognition and detection model."
        )
    )
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=[*SUPPORTED_TASKS, "all"],
        default=["all"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["validation", "test"],
        default=None,
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--checkpoint-policy",
        choices=["best_only", "best_then_last"],
        default=None,
    )
    parser.add_argument("--recognition-batch-size", type=int, default=None)
    parser.add_argument("--detection-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan_path = _project_path(args.plan)
    plan = load_yaml(plan_path)

    selected_tasks = list(SUPPORTED_TASKS) if "all" in args.tasks else list(args.tasks)
    selected_splits = list(args.splits or plan["protocol"]["default_splits"])
    selected_splits = [normalize_split_name(split) for split in selected_splits]
    checkpoint_policy = str(
        args.checkpoint_policy or plan["protocol"]["checkpoint_policy"]
    )
    output_root = resolve_output_root(plan, selected_tasks, args.output_root)

    evaluations, issues = build_model_evaluations(
        plan=plan,
        selected_tasks=selected_tasks,
        selected_splits=selected_splits,
        output_root=output_root,
        checkpoint_policy=checkpoint_policy,
        model_filter=None if not args.models else set(args.models),
        batch_overrides={
            "recognition": args.recognition_batch_size,
            "detection": args.detection_batch_size,
        },
        num_workers_override=args.num_workers,
    )
    issues = list(dict.fromkeys(issues))

    expected_model_count = sum(
        len(configured_model_paths(plan, task)) for task in selected_tasks
    )
    if args.models:
        expected_model_count = len(set(args.models))

    print(f"[PLAN] target={plan['target_runtime']['hardware']}")
    print(f"[PLAN] tasks={','.join(selected_tasks)}")
    print(f"[PLAN] splits={','.join(selected_splits)}")
    print(f"[PLAN] checkpoint_policy={checkpoint_policy}")
    print(f"[PLAN] output_root={output_root}")
    print(f"[PLAN] models_ready={len(evaluations)}/{expected_model_count}")

    if issues:
        for issue in issues:
            print(f"[MISSING] {issue}", file=sys.stderr)
        if not args.allow_missing:
            print(
                "[ERROR] Preflight failed. No held-out evaluation was started. "
                "Use --allow-missing only for an explicitly partial run.",
                file=sys.stderr,
            )
            return 2

    if not evaluations:
        print("[ERROR] No runnable model evaluations were found.", file=sys.stderr)
        return 2

    commands = [
        (evaluation, split, build_evaluation_command(
            evaluation,
            split=split,
            python_bin=args.python_bin,
        ))
        for evaluation in evaluations
        for split in selected_splits
        if evaluation.manifests[split].is_file()
    ]

    for evaluation, split, command in commands:
        print(
            f"[COMMAND] task={evaluation.task} model={evaluation.experiment_name} "
            f"split={split} checkpoint={evaluation.checkpoint_path} "
            f"manifest={evaluation.manifests[split]}"
        )
        print(f"[COMMAND] {shlex.join(command)}")

    if args.dry_run:
        print(f"[OK] Dry run validated {len(commands)} evaluation command(s).")
        return 0

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SRC_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    environment.setdefault("OCR_MANCHU_PROJECT_ROOT", str(PROJECT_ROOT))
    environment["OCR_MANCHU_OUTPUT_ROOT"] = str(output_root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_dir = output_root / str(
        plan.get("outputs", {}).get(
            "run_summary_subdir",
            "metrics/experiment_runs",
        )
    )
    summary_path = summary_dir / f"heldout_all_models_{timestamp}.json"
    summary: dict[str, Any] = {
        "schema_version": "ocr_manchu.heldout_run.v1",
        "experiment_name": str(plan["experiment"]["name"]),
        "target_hardware": str(plan["target_runtime"]["hardware"]),
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "tasks": selected_tasks,
        "splits": selected_splits,
        "checkpoint_policy": checkpoint_policy,
        "output_root": normalize_result_path(output_root),
        "planned_runs": len(commands),
        "runs": [],
    }
    failures = 0

    for evaluation, split, command in commands:
        started = perf_counter()
        print(
            f"[RUN] task={evaluation.task} model={evaluation.experiment_name} "
            f"split={split} checkpoint_tag={evaluation.checkpoint_tag}"
        )
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        process_seconds = round(max(0.0, perf_counter() - started), 6)
        result_path = result_path_for(output_root, evaluation, split)
        run_record: dict[str, Any] = {
            "task": evaluation.task,
            "model_name": evaluation.experiment_name,
            "model_architecture": evaluation.model_architecture,
            "split": split,
            "checkpoint_path": normalize_result_path(evaluation.checkpoint_path),
            "checkpoint_tag": evaluation.checkpoint_tag,
            "manifest_path": normalize_result_path(evaluation.manifests[split]),
            "result_path": normalize_result_path(result_path),
            "process_runtime_seconds": process_seconds,
            "return_code": int(completed.returncode),
            "status": "completed" if completed.returncode == 0 else "failed",
        }

        if completed.returncode == 0:
            try:
                result = validate_result_file(result_path, evaluation, split)
                run_record["metrics"] = result["metrics"]
                run_record["runtime_seconds"] = result["runtime_seconds"]
            except Exception as exc:
                run_record["status"] = "invalid_result"
                run_record["error"] = str(exc)
                failures += 1
        else:
            failures += 1

        summary["runs"].append(run_record)
        summary["finished_at_utc"] = utc_now()
        _write_run_summary(summary_path, summary)

        if run_record["status"] != "completed" and args.fail_fast:
            break

    summary["finished_at_utc"] = utc_now()
    summary["completed_runs"] = sum(
        1 for item in summary["runs"] if item["status"] == "completed"
    )
    summary["failed_runs"] = failures
    _write_run_summary(summary_path, summary)

    print(f"[RESULT] completed={summary['completed_runs']} failed={failures}")
    print(f"[RESULT] run_summary={summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
