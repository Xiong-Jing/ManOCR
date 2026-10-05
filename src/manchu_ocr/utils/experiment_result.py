from __future__ import annotations

import json
import platform
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping


RESULT_SCHEMA_VERSION = "ocr_manchu.experiment_result.v1"


def normalize_split_name(split: str) -> str:
    """Return the public split name used in experiment result files."""
    normalized = split.strip().lower()
    aliases = {
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "train": "train",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported split: {split!r}")
    return aliases[normalized]


def split_key(split: str) -> str:
    """Return the short split key used in manifest and output filenames."""
    public_name = normalize_split_name(split)
    return "val" if public_name == "validation" else public_name


def normalize_result_path(path: str | Path) -> str:
    """Normalize local or remote paths without resolving them on this host."""
    return str(path).replace("\\", "/")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ExperimentTiming:
    started_at_utc: str
    finished_at_utc: str
    elapsed_seconds: float


class ExperimentTimer:
    """Wall-clock timer for one validation or held-out test process."""

    def __init__(self) -> None:
        self.started_at_utc = utc_now()
        self._started_at = perf_counter()

    def finish(self) -> ExperimentTiming:
        return ExperimentTiming(
            started_at_utc=self.started_at_utc,
            finished_at_utc=utc_now(),
            elapsed_seconds=max(0.0, perf_counter() - self._started_at),
        )


def collect_torch_runtime(device: Any) -> dict[str, Any]:
    """Collect reproducibility metadata without making CUDA a hard dependency."""
    import torch

    details: dict[str, Any] = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
    }

    if str(device).startswith("cuda") and torch.cuda.is_available():
        device_index = getattr(device, "index", None)
        if device_index is None:
            device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        details.update(
            {
                "gpu_index": int(device_index),
                "gpu_name": str(properties.name),
                "gpu_total_memory_gb": round(
                    float(properties.total_memory) / (1024**3),
                    3,
                ),
                "peak_gpu_allocated_mb": round(
                    float(torch.cuda.max_memory_allocated(device_index)) / (1024**2),
                    3,
                ),
                "peak_gpu_reserved_mb": round(
                    float(torch.cuda.max_memory_reserved(device_index)) / (1024**2),
                    3,
                ),
            }
        )

    return details


def build_experiment_result(
    *,
    task: str,
    experiment_name: str,
    model_name: str,
    model_architecture: str,
    split: str,
    config_path: str | Path,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    metrics: Mapping[str, Any],
    timing: ExperimentTiming,
    runtime_details: Mapping[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common result envelope used by every evaluation script."""
    public_split = normalize_split_name(split)
    short_split = split_key(split)
    elapsed_seconds = round(float(timing.elapsed_seconds), 6)

    runtime = {
        "started_at_utc": timing.started_at_utc,
        "finished_at_utc": timing.finished_at_utc,
        "elapsed_seconds": elapsed_seconds,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    if runtime_details:
        runtime.update(dict(runtime_details))

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": task,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "model_architecture": model_architecture,
        "split": public_split,
        "split_key": short_split,
        "config_path": normalize_result_path(config_path),
        "checkpoint_path": normalize_result_path(checkpoint_path),
        "manifest_path": normalize_result_path(manifest_path),
        "metrics": dict(metrics),
        "runtime_seconds": elapsed_seconds,
        "runtime": runtime,
    }

    if extra_fields:
        collisions = sorted(set(result).intersection(extra_fields))
        if collisions:
            raise ValueError(
                "extra_fields cannot replace standard result keys: "
                + ", ".join(collisions)
            )
        result.update(dict(extra_fields))

    return result


def format_experiment_result(result: Mapping[str, Any]) -> str:
    """Format one complete, machine-searchable result line for logs."""
    metrics_text = json.dumps(
        result["metrics"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "RESULT "
        f"model_name={result['model_name']} "
        f"split={result['split']} "
        f"checkpoint_path={result['checkpoint_path']} "
        f"manifest_path={result['manifest_path']} "
        f"runtime_seconds={result['runtime_seconds']} "
        f"metrics={metrics_text}"
    )
