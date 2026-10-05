#!/usr/bin/env python3
"""Run zero-shot Manchu word OCR through hosted multimodal APIs.

Ground-truth labels are loaded locally only after each response is returned;
they are never included in an API payload. A final result is emitted only when
every selected sample has a successful API response.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manchu_ocr.api_eval.providers import (  # noqa: E402
    ProviderConfigurationError,
    ProviderRequestError,
    create_provider,
    public_endpoint,
    validate_provider_environment,
)
from manchu_ocr.api_eval.zero_shot import (  # noqa: E402
    RecognitionSample,
    aggregate_strict_metrics,
    classify_error_mode,
    deterministic_subset,
    length_bin_counts,
    load_recognition_manifest,
    manifest_sha256,
    normalize_romanized_manchu,
    sample_error_counts,
    text_sha256,
)
from manchu_ocr.utils.config import load_yaml  # noqa: E402
from manchu_ocr.utils.experiment_result import (  # noqa: E402
    ExperimentTiming,
    build_experiment_result,
    format_experiment_result,
    normalize_result_path,
    normalize_split_name,
    split_key,
    utc_now,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "lmm_zero_shot.yaml"
RUN_SCHEMA = "ocr_manchu.lmm_zero_shot_run.v1"
RECORD_SCHEMA = "ocr_manchu.lmm_zero_shot_prediction.v1"


@dataclass(frozen=True)
class PreparedSplit:
    public_name: str
    source_manifest: Path
    evaluation_manifest: Path
    source_samples: tuple[RecognitionSample, ...]
    selected_samples: tuple[RecognitionSample, ...]
    source_sha256: str
    selected_sha256: str
    full_split: bool


@dataclass(frozen=True)
class RunPaths:
    record_jsonl: Path
    predictions_csv: Path
    result_json: Path
    status_json: Path
    metadata_json: Path


@dataclass(frozen=True)
class RunOutcome:
    model_key: str
    display_name: str
    provider: str
    model_identifier: str
    split: str
    status: str
    selected_samples: int
    result_path: Path
    status_path: Path
    result: Mapping[str, Any] | None


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._-")
    if not slug:
        raise ValueError(f"Cannot construct a safe path component from {value!r}")
    return slug


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _json_fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text_sha256(canonical)


def _selected_manifest_sha256(samples: Sequence[RecognitionSample]) -> str:
    material = "".join(f"{item.image_path}\t{item.label}\n" for item in samples)
    return text_sha256(material)


def _write_selected_manifest(
    *,
    path: Path,
    samples: Sequence[RecognitionSample],
    source_manifest: Path,
    source_sha256: str,
    seed: int,
    split: str,
    overwrite: bool,
) -> None:
    expected = "".join(f"{item.image_path}\t{item.label}\n" for item in samples)
    if path.exists() and path.read_text(encoding="utf-8") != expected and not overwrite:
        raise RuntimeError(
            f"Existing selected manifest does not match this run: {path}. "
            "Use a new --run-tag or --overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")
    metadata = {
        "schema_version": "ocr_manchu.lmm_selected_manifest.v1",
        "split": split,
        "source_manifest": normalize_result_path(source_manifest),
        "source_manifest_sha256": source_sha256,
        "selection_seed": seed,
        "selection_method": "deterministic_length_stratified_sha256",
        "selected_samples": len(samples),
        "selected_length_bins": length_bin_counts(samples),
        "selected_manifest_sha256": _selected_manifest_sha256(samples),
    }
    _atomic_json(path.with_suffix(path.suffix + ".metadata.json"), metadata)


def _resolve_output_root(
    paths_config: Mapping[str, Any], override: str | None
) -> Path:
    value = override or os.environ.get("OCR_MANCHU_OUTPUT_ROOT")
    if value is None:
        value = str(paths_config["outputs"]["root"])
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def prepare_splits(
    *,
    experiment_config: Mapping[str, Any],
    paths_config: Mapping[str, Any],
    selected_splits: Sequence[str],
    output_root: Path,
    run_tag: str,
    max_samples: int | None,
    skip_image_check: bool,
    overwrite: bool,
) -> dict[str, PreparedSplit]:
    seed = int(experiment_config["protocol"].get("selection_seed", 42))
    recognition = paths_config["recognition_data"]
    prepared: dict[str, PreparedSplit] = {}
    manifests_root = (
        output_root
        / "metrics"
        / "lmm_zero_shot"
        / _slug(str(experiment_config["experiment"]["protocol_version"]))
        / _slug(run_tag)
        / "manifests"
    )

    for split in selected_splits:
        public_name = normalize_split_name(split)
        source_manifest = Path(
            recognition["val_list" if public_name == "validation" else "test_list"]
        )
        source_samples = load_recognition_manifest(source_manifest)
        selected = deterministic_subset(
            source_samples,
            limit=max_samples,
            seed=seed,
            split=public_name,
        )
        full_split = len(selected) == len(source_samples)
        source_hash = manifest_sha256(source_manifest)

        if full_split:
            evaluation_manifest = source_manifest
            selected_hash = source_hash
        else:
            evaluation_manifest = manifests_root / f"{split_key(public_name)}.txt"
            _write_selected_manifest(
                path=evaluation_manifest,
                samples=selected,
                source_manifest=source_manifest,
                source_sha256=source_hash,
                seed=seed,
                split=public_name,
                overwrite=overwrite,
            )
            selected_hash = _selected_manifest_sha256(selected)

        if not skip_image_check:
            missing = [item.image_path for item in selected if not Path(item.image_path).is_file()]
            if missing:
                examples = "\n".join(missing[:5])
                raise FileNotFoundError(
                    f"{len(missing)} selected images are missing for {public_name}. "
                    f"First examples:\n{examples}"
                )

        prepared[public_name] = PreparedSplit(
            public_name=public_name,
            source_manifest=source_manifest,
            evaluation_manifest=evaluation_manifest,
            source_samples=tuple(source_samples),
            selected_samples=tuple(selected),
            source_sha256=source_hash,
            selected_sha256=selected_hash,
            full_split=full_split,
        )
    return prepared


def _model_identifier(config: Mapping[str, Any]) -> str:
    return str(config.get("model_id") or config.get("source") or "unknown")


def _checkpoint_uri(config: Mapping[str, Any]) -> str:
    provider = _slug(str(config["provider"]))
    model = _model_identifier(config).strip().replace(" ", "_")
    return f"api://{provider}/{model}"


def _run_paths(
    *,
    output_root: Path,
    protocol_version: str,
    run_tag: str,
    model_key: str,
    split: str,
) -> RunPaths:
    common = Path(
        "lmm_zero_shot",
        _slug(protocol_version),
        _slug(run_tag),
        _slug(model_key),
    )
    metrics_dir = output_root / "metrics" / common
    predictions_dir = output_root / "predictions" / common
    short = split_key(split)
    return RunPaths(
        record_jsonl=predictions_dir / f"predictions_{short}.jsonl",
        predictions_csv=predictions_dir / f"predictions_{short}.csv",
        result_json=metrics_dir / f"eval_{short}.json",
        status_json=metrics_dir / f"status_{short}.json",
        metadata_json=predictions_dir / f"run_{short}.json",
    )


def _load_record_history(path: Path, fingerprint: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL record at {path}:{line_number}; "
                    "use --overwrite after preserving the damaged file"
                ) from exc
            if record.get("run_fingerprint") != fingerprint:
                raise ValueError(
                    f"Fingerprint mismatch inside {path}:{line_number}; "
                    "use a new --run-tag or --overwrite"
                )
            records.append(record)
    return records


def _load_latest_records(path: Path, fingerprint: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in _load_record_history(path, fingerprint):
        latest[str(record["sample_id"])] = record
    return latest


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def _normalization_kwargs(config: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "remove_whitespace": bool(config.get("remove_whitespace", True)),
        "lowercase": bool(config.get("lowercase", True)),
        "normalize_apostrophes": bool(config.get("normalize_apostrophes", True)),
    }


def _applied_generation(
    model_config: Mapping[str, Any], common_generation: Mapping[str, Any]
) -> dict[str, Any]:
    provider = str(model_config["provider"])
    if provider == "openai_compatible":
        return dict(common_generation)
    if provider == "paddleocr_official":
        options = dict(model_config.get("options", {}))
        return {
            key: options[key]
            for key in ("temperature", "top_p", "max_new_tokens")
            if key in options
        }
    # The official GOT Gradio task exposes no decoding controls.
    return {}


def _request_one(
    *,
    sample: RecognitionSample,
    provider_factory: Any,
    thread_local: threading.local,
    providers: list[Any],
    providers_lock: threading.Lock,
    retry_config: Mapping[str, Any],
    normalization: Mapping[str, bool],
    allowed_characters: set[str],
    fingerprint: str,
    model_key: str,
    split: str,
) -> dict[str, Any]:
    total_started = perf_counter()
    max_attempts = max(1, int(retry_config.get("max_attempts", 5)))
    initial_backoff = max(0.0, float(retry_config.get("initial_backoff_seconds", 2.0)))
    max_backoff = min(60.0, max(0.0, float(retry_config.get("max_backoff_seconds", 30.0))))
    jitter = max(0.0, float(retry_config.get("jitter_seconds", 0.5)))
    request_delay = min(60.0, max(0.0, float(retry_config.get("request_delay_seconds", 0.0))))
    last_error: Exception | None = None
    last_retryable = False
    attempts = 0

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            provider = getattr(thread_local, "provider", None)
            if provider is None:
                provider = provider_factory()
                thread_local.provider = provider
                with providers_lock:
                    providers.append(provider)
            if request_delay:
                time.sleep(request_delay)
            output = provider.predict(sample.image_path)
            raw_response = str(output.text)
            reference = normalize_romanized_manchu(sample.label, **normalization)
            prediction = normalize_romanized_manchu(raw_response, **normalization)
            counts = sample_error_counts(reference, prediction)
            return {
                "schema_version": RECORD_SCHEMA,
                "run_fingerprint": fingerprint,
                "model": model_key,
                "split": split,
                "sample_id": sample.sample_id,
                "source_index": sample.source_index,
                "image_path": sample.image_path,
                "ground_truth": reference,
                "raw_response": raw_response,
                "prediction": prediction,
                "status": "success",
                "S": counts["S"],
                "D": counts["D"],
                "I": counts["I"],
                "N": counts["N"],
                "edit_distance": counts["edit_distance"],
                "error_mode": classify_error_mode(
                    reference=reference,
                    prediction=prediction,
                    raw_response=raw_response,
                    allowed_characters=allowed_characters,
                    counts=counts,
                ),
                "attempts": attempt,
                "latency_seconds": round(perf_counter() - total_started, 6),
                "provider_metadata": output.metadata,
                "error": None,
                "recorded_at_utc": utc_now(),
            }
        except ProviderRequestError as exc:
            last_error = exc
            last_retryable = bool(exc.retryable)
            if not exc.retryable or attempt >= max_attempts:
                break
            exponential = min(max_backoff, initial_backoff * (2 ** (attempt - 1)))
            delay = max(exponential, float(exc.retry_after_seconds or 0.0))
            delay = min(60.0, delay + random.uniform(0.0, jitter))
            if delay:
                time.sleep(delay)
        except ProviderConfigurationError as exc:
            last_error = exc
            last_retryable = False
            break
        except Exception as exc:  # provider boundary: preserve run rather than crash
            last_error = exc
            last_retryable = False
            break

    return {
        "schema_version": RECORD_SCHEMA,
        "run_fingerprint": fingerprint,
        "model": model_key,
        "split": split,
        "sample_id": sample.sample_id,
        "source_index": sample.source_index,
        "image_path": sample.image_path,
        "ground_truth": normalize_romanized_manchu(sample.label, **normalization),
        "raw_response": "",
        "prediction": "",
        "status": "api_error",
        "S": None,
        "D": None,
        "I": None,
        "N": len(normalize_romanized_manchu(sample.label, **normalization)),
        "edit_distance": None,
        "error_mode": "api_error",
        "attempts": attempts,
        "latency_seconds": round(perf_counter() - total_started, 6),
        "provider_metadata": {},
        "error": f"{type(last_error).__name__}: {last_error}"[:2000],
        "retryable": last_retryable,
        "recorded_at_utc": utc_now(),
    }


def _run_pending_requests(
    *,
    samples: Sequence[RecognitionSample],
    model_config: Mapping[str, Any],
    prompt: str,
    generation: Mapping[str, Any],
    retry_config: Mapping[str, Any],
    normalization: Mapping[str, bool],
    allowed_characters: set[str],
    fingerprint: str,
    model_key: str,
    split: str,
    workers: int,
    max_errors: int,
    record_path: Path,
    progress_every: int,
) -> tuple[int, bool]:
    thread_local = threading.local()
    providers: list[Any] = []
    providers_lock = threading.Lock()
    timeout = float(retry_config.get("request_timeout_seconds", 180.0))

    def provider_factory() -> Any:
        return create_provider(
            model_config,
            prompt=prompt,
            generation=generation,
            timeout_seconds=timeout,
        )

    def submit_sample(executor: ThreadPoolExecutor, sample: RecognitionSample) -> Future:
        return executor.submit(
            _request_one,
            sample=sample,
            provider_factory=provider_factory,
            thread_local=thread_local,
            providers=providers,
            providers_lock=providers_lock,
            retry_config=retry_config,
            normalization=normalization,
            allowed_characters=allowed_characters,
            fingerprint=fingerprint,
            model_key=model_key,
            split=split,
        )

    completed = 0
    errors = 0
    stopped_early = False
    iterator = iter(samples)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lmm-api")
    active: dict[Future, RecognitionSample] = {}

    def persist_finished(futures: Iterable[Future]) -> None:
        nonlocal completed, errors
        for future in futures:
            active.pop(future, None)
            if future.cancelled():
                continue
            record = future.result()
            _append_record(record_path, record)
            completed += 1
            errors += int(record["status"] != "success")
            if completed % progress_every == 0 or completed == len(samples):
                print(
                    f"[PROGRESS] model={model_key} split={split} "
                    f"completed={completed}/{len(samples)} api_errors={errors}",
                    flush=True,
                )

    try:
        for _ in range(min(len(samples), workers)):
            try:
                sample = next(iterator)
            except StopIteration:
                break
            active[submit_sample(executor, sample)] = sample

        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            persist_finished(done)

            if errors >= max_errors:
                stopped_early = True
                print(
                    f"[STOP] model={model_key} split={split} reached "
                    f"max_errors={max_errors}; rerun with resume after fixing the API",
                    file=sys.stderr,
                )
                break

            while len(active) < workers:
                try:
                    sample = next(iterator)
                except StopIteration:
                    break
                active[submit_sample(executor, sample)] = sample
    finally:
        if stopped_early:
            for future in active:
                future.cancel()
            running = [future for future in active if not future.cancelled()]
            if running:
                finished, _ = wait(running)
                # Cache every request that reached the provider, even after the
                # circuit breaker fired, so a paid successful call is not lost.
                persist_finished(finished)
        executor.shutdown(wait=True, cancel_futures=True)
        for provider in providers:
            try:
                provider.close()
            except Exception:
                pass
    return errors, stopped_early


def _usage_totals(records: Iterable[Mapping[str, Any]]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for record in records:
        metadata = record.get("provider_metadata")
        if not isinstance(metadata, Mapping):
            continue
        usage = metadata.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[str(key)] = totals.get(str(key), 0) + value
    return totals


def _write_predictions_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "split",
        "sample_id",
        "source_index",
        "image_path",
        "ground_truth",
        "prediction",
        "raw_response",
        "status",
        "S",
        "D",
        "I",
        "N",
        "edit_distance",
        "error_mode",
        "attempts",
        "latency_seconds",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _endpoint_description(config: Mapping[str, Any]) -> str:
    if config["provider"] == "openai_compatible":
        return public_endpoint(str(config["base_url"]))
    if config["provider"] == "gradio_space":
        return str(config["source"])
    base_url_env = str(config.get("base_url_env", "PADDLEOCR_BASE_URL"))
    custom = os.environ.get(base_url_env)
    return public_endpoint(custom) if custom else "PaddleOCR official hosted API"


def _write_status(
    path: Path,
    *,
    model_key: str,
    display_name: str,
    split: PreparedSplit,
    fingerprint: str,
    status: str,
    records: Mapping[str, Mapping[str, Any]],
    result_path: Path,
) -> dict[str, Any]:
    successful = sum(item.get("status") == "success" for item in records.values())
    api_errors = sum(item.get("status") == "api_error" for item in records.values())
    value = {
        "schema_version": RUN_SCHEMA,
        "model": model_key,
        "display_name": display_name,
        "split": split.public_name,
        "status": status,
        "run_fingerprint": fingerprint,
        "source_manifest": normalize_result_path(split.source_manifest),
        "manifest_path": normalize_result_path(split.evaluation_manifest),
        "selected_samples": len(split.selected_samples),
        "successful_samples": successful,
        "api_error_samples": api_errors,
        "missing_samples": len(split.selected_samples) - successful - api_errors,
        "result_path": normalize_result_path(result_path),
        "updated_at_utc": utc_now(),
        "message": (
            "Complete and reportable"
            if status == "completed"
            else "Incomplete: fix API errors and rerun with resume; do not report metrics"
        ),
    }
    _atomic_json(path, value)
    return value


def run_model_split(
    *,
    experiment_config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    run_tag: str,
    model_key: str,
    model_config: Mapping[str, Any],
    prepared_split: PreparedSplit,
    workers_override: int | None,
    max_errors_override: int | None,
    overwrite: bool,
) -> RunOutcome:
    protocol = experiment_config["protocol"]
    prompt = str(protocol["prompt"])
    generation = dict(protocol.get("generation", {}))
    applied_generation = _applied_generation(model_config, generation)
    retry_config = dict(protocol.get("retry", {}))
    normalization = _normalization_kwargs(protocol.get("normalization", {}))
    display_name = str(model_config["display_name"])
    provider_name = str(model_config["provider"])
    identifier = _model_identifier(model_config)
    paths = _run_paths(
        output_root=output_root,
        protocol_version=str(experiment_config["experiment"]["protocol_version"]),
        run_tag=run_tag,
        model_key=model_key,
        split=prepared_split.public_name,
    )
    workers = max(1, int(workers_override or model_config.get("workers", 1)))
    max_errors = max(
        1,
        int(max_errors_override or retry_config.get("max_errors_before_stop", 5)),
    )
    fingerprint_payload = {
        "schema": RUN_SCHEMA,
        "protocol_version": experiment_config["experiment"]["protocol_version"],
        "model_key": model_key,
        "provider": provider_name,
        "model_identifier": identifier,
        "endpoint": _endpoint_description(model_config),
        "split": prepared_split.public_name,
        "selected_manifest_sha256": prepared_split.selected_sha256,
        "prompt": prompt if bool(model_config.get("accepts_text_prompt", False)) else None,
        "model_native_task": model_config.get("model_native_task"),
        "generation": applied_generation,
        "normalization": normalization,
    }
    fingerprint = _json_fingerprint(fingerprint_payload)

    generated_paths = [
        paths.record_jsonl,
        paths.predictions_csv,
        paths.result_json,
        paths.status_json,
        paths.metadata_json,
    ]
    if overwrite:
        for path in generated_paths:
            if path.is_file():
                path.unlink()

    metadata: dict[str, Any]
    if paths.metadata_json.is_file():
        metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        if metadata.get("run_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing run has a different fingerprint: {paths.metadata_json}. "
                "Use a new --run-tag or --overwrite."
            )
    else:
        metadata = {
            "schema_version": RUN_SCHEMA,
            "run_fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "started_at_utc": utc_now(),
            "active_wall_seconds": 0.0,
            "invocations": 0,
        }
        _atomic_json(paths.metadata_json, metadata)

    latest = _load_latest_records(paths.record_jsonl, fingerprint)
    selected_ids = {item.sample_id for item in prepared_split.selected_samples}
    unknown_ids = sorted(set(latest).difference(selected_ids))
    if unknown_ids:
        raise RuntimeError(
            f"Prediction cache contains {len(unknown_ids)} samples outside the "
            "selected manifest; use a new --run-tag or --overwrite"
        )
    pending = [
        sample
        for sample in prepared_split.selected_samples
        if latest.get(sample.sample_id, {}).get("status") != "success"
    ]

    print(
        f"[RUN] model={display_name} split={prepared_split.public_name} "
        f"checkpoint={_checkpoint_uri(model_config)} "
        f"manifest={prepared_split.evaluation_manifest} "
        f"selected={len(prepared_split.selected_samples)} resume_success={len(prepared_split.selected_samples) - len(pending)} "
        f"pending={len(pending)} workers={workers}",
        flush=True,
    )

    active_started = perf_counter()
    if pending:
        if paths.result_json.is_file():
            paths.result_json.unlink()
        allowed_characters = {
            character
            for item in prepared_split.source_samples
            for character in normalize_romanized_manchu(item.label, **normalization)
        }
        progress_every = max(1, int(protocol.get("progress_every", 25)))
        _run_pending_requests(
            samples=pending,
            model_config=model_config,
            prompt=prompt,
            generation=generation,
            retry_config=retry_config,
            normalization=normalization,
            allowed_characters=allowed_characters,
            fingerprint=fingerprint,
            model_key=model_key,
            split=prepared_split.public_name,
            workers=workers,
            max_errors=max_errors,
            record_path=paths.record_jsonl,
            progress_every=progress_every,
        )

    invocation_seconds = max(0.0, perf_counter() - active_started)
    metadata["active_wall_seconds"] = round(
        float(metadata.get("active_wall_seconds", 0.0)) + invocation_seconds,
        6,
    )
    metadata["invocations"] = int(metadata.get("invocations", 0)) + 1
    metadata["updated_at_utc"] = utc_now()
    _atomic_json(paths.metadata_json, metadata)

    latest = _load_latest_records(paths.record_jsonl, fingerprint)
    selected_records = {
        sample.sample_id: latest[sample.sample_id]
        for sample in prepared_split.selected_samples
        if sample.sample_id in latest
    }
    successful = {
        sample_id: record
        for sample_id, record in selected_records.items()
        if record.get("status") == "success"
    }
    complete = len(successful) == len(prepared_split.selected_samples)
    if not complete:
        _write_status(
            paths.status_json,
            model_key=model_key,
            display_name=display_name,
            split=prepared_split,
            fingerprint=fingerprint,
            status="incomplete",
            records=selected_records,
            result_path=paths.result_json,
        )
        return RunOutcome(
            model_key=model_key,
            display_name=display_name,
            provider=provider_name,
            model_identifier=identifier,
            split=prepared_split.public_name,
            status="incomplete",
            selected_samples=len(prepared_split.selected_samples),
            result_path=paths.result_json,
            status_path=paths.status_json,
            result=None,
        )

    ordered = [successful[item.sample_id] for item in prepared_split.selected_samples]
    history = _load_record_history(paths.record_jsonl, fingerprint)
    references = [str(record["ground_truth"]) for record in ordered]
    predictions = [str(record["prediction"]) for record in ordered]
    metrics = aggregate_strict_metrics(references, predictions)
    latencies = [float(record["latency_seconds"]) for record in ordered]
    cumulative_attempts: dict[str, int] = {}
    for record in history:
        sample_id = str(record["sample_id"])
        cumulative_attempts[sample_id] = cumulative_attempts.get(sample_id, 0) + int(
            record.get("attempts", 0)
        )
    history_latencies = [float(record.get("latency_seconds", 0.0)) for record in history]
    request_statistics = {
        "successful_samples": len(ordered),
        "api_error_samples": 0,
        "recorded_request_cycles": len(history),
        "total_attempts": sum(cumulative_attempts.values()),
        "retried_samples": sum(value > 1 for value in cumulative_attempts.values()),
        "total_request_latency_seconds": round(sum(history_latencies), 6),
        "mean_final_success_latency_seconds": round(sum(latencies) / len(latencies), 6),
        "max_final_success_latency_seconds": round(max(latencies), 6),
        "usage": _usage_totals(history),
    }
    error_modes: dict[str, int] = {}
    for record in ordered:
        mode = str(record["error_mode"])
        error_modes[mode] = error_modes.get(mode, 0) + 1

    _write_predictions_csv(paths.predictions_csv, ordered)
    timing = ExperimentTiming(
        started_at_utc=str(metadata["started_at_utc"]),
        finished_at_utc=utc_now(),
        elapsed_seconds=float(metadata["active_wall_seconds"]),
    )
    result = build_experiment_result(
        task="recognition",
        experiment_name=str(experiment_config["experiment"]["name"]),
        model_name=model_key,
        model_architecture=display_name,
        split=prepared_split.public_name,
        config_path=config_path,
        checkpoint_path=_checkpoint_uri(model_config),
        manifest_path=prepared_split.evaluation_manifest,
        metrics=metrics,
        timing=timing,
        runtime_details={
            "execution_mode": "hosted_api_zero_shot",
            "workers": workers,
            "python_implementation": platform.python_implementation(),
        },
        extra_fields={
            "display_name": display_name,
            "api": {
                "provider": provider_name,
                "model_identifier": identifier,
                "endpoint": _endpoint_description(model_config),
                "no_local_checkpoint": True,
            },
            "protocol": {
                "zero_shot": True,
                "training_performed": False,
                "input_unit": "recognition_word_crop",
                "prompt": prompt if bool(model_config.get("accepts_text_prompt", False)) else None,
                "prompt_sha256": (
                    text_sha256(prompt)
                    if bool(model_config.get("accepts_text_prompt", False))
                    else None
                ),
                "configured_common_prompt_sha256": text_sha256(prompt),
                "model_native_task": model_config.get("model_native_task"),
                "generation": applied_generation,
                "normalization": normalization,
                "strict_common_metrics": True,
                "word_tolerance": 0,
                "character_tolerance": 0,
                "spaces_count_as_characters": not normalization["remove_whitespace"],
                "full_split": prepared_split.full_split,
                "run_tag": run_tag,
                "run_fingerprint": fingerprint,
            },
            "source_manifest_path": normalize_result_path(prepared_split.source_manifest),
            "source_manifest_sha256": prepared_split.source_sha256,
            "evaluation_manifest_sha256": prepared_split.selected_sha256,
            "request_statistics": request_statistics,
            "failure_mode_counts": error_modes,
            "artifacts": {
                "prediction_records_jsonl": normalize_result_path(paths.record_jsonl),
                "predictions_csv": normalize_result_path(paths.predictions_csv),
                "run_metadata": normalize_result_path(paths.metadata_json),
            },
        },
    )
    _atomic_json(paths.result_json, result)
    _write_status(
        paths.status_json,
        model_key=model_key,
        display_name=display_name,
        split=prepared_split,
        fingerprint=fingerprint,
        status="completed",
        records=selected_records,
        result_path=paths.result_json,
    )
    print(format_experiment_result(result), flush=True)
    return RunOutcome(
        model_key=model_key,
        display_name=display_name,
        provider=provider_name,
        model_identifier=identifier,
        split=prepared_split.public_name,
        status="completed",
        selected_samples=len(prepared_split.selected_samples),
        result_path=paths.result_json,
        status_path=paths.status_json,
        result=result,
    )


def write_summary(
    *,
    outcomes: Sequence[RunOutcome],
    output_root: Path,
    protocol_version: str,
    run_tag: str,
) -> tuple[Path, Path, Path]:
    root = (
        output_root
        / "metrics"
        / "lmm_zero_shot"
        / _slug(protocol_version)
        / _slug(run_tag)
    )
    new_rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        result = outcome.result or {}
        metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}
        new_rows.append(
            {
                "model": outcome.display_name,
                "split": outcome.split,
                "provider": outcome.provider,
                "model_id": outcome.model_identifier,
                "status": outcome.status,
                "samples": outcome.selected_samples,
                "S": metrics.get("S"),
                "D": metrics.get("D"),
                "I": metrics.get("I"),
                "N": metrics.get("N"),
                "CER": metrics.get("CER"),
                "WA": metrics.get("WA"),
                "CA": metrics.get("CA"),
                "checkpoint_path": result.get("checkpoint_path"),
                "manifest_path": result.get("manifest_path"),
                "runtime_seconds": result.get("runtime_seconds"),
                "result_json": normalize_result_path(outcome.result_path),
                "status_json": normalize_result_path(outcome.status_path),
            }
        )

    columns = [
        "model",
        "split",
        "S",
        "D",
        "I",
        "N",
        "CER",
        "WA",
        "CA",
        "provider",
        "model_id",
        "status",
        "samples",
        "checkpoint_path",
        "manifest_path",
        "runtime_seconds",
        "result_json",
        "status_json",
    ]
    csv_path = root / "lmm_zero_shot_summary.csv"
    json_path = root / "lmm_zero_shot_summary.json"
    markdown_path = root / "lmm_zero_shot_summary.md"
    root.mkdir(parents=True, exist_ok=True)

    # Validation and test are intentionally allowed to run as separate jobs.
    # Merge their rows so the second command does not erase the first summary.
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    if json_path.is_file():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
            for row in existing.get("rows", []):
                if isinstance(row, Mapping) and row.get("model") and row.get("split"):
                    merged[(str(row["model"]), str(row["split"]))] = dict(row)
        except (json.JSONDecodeError, OSError, AttributeError):
            pass
    for row in new_rows:
        merged[(str(row["model"]), str(row["split"]))] = row
    split_order = {"validation": 0, "test": 1}
    rows = sorted(
        merged.values(),
        key=lambda row: (
            split_order.get(str(row["split"]), 99),
            str(row["model"]),
        ),
    )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(
        json_path,
        {
            "schema_version": "ocr_manchu.lmm_zero_shot_summary.v1",
            "protocol_version": protocol_version,
            "run_tag": run_tag,
            "generated_at_utc": utc_now(),
            "rows": rows,
        },
    )

    def percent(value: Any) -> str:
        return "-" if value is None else f"{float(value) * 100:.2f}"

    lines = [
        "# Zero-shot OCR API results",
        "",
        "WA is exact word accuracy. CA is retained-reference character accuracy. "
        "CER is corpus edit-distance CER, `(S + D + I) / N`. All are strict and "
        "use zero tolerance.",
        "",
        "| Model | Split | Status | Samples | S | D | I | N | CER (%) | WA (%) | CA (%) | Runtime (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        runtime = "-" if row["runtime_seconds"] is None else f"{float(row['runtime_seconds']):.2f}"
        lines.append(
            f"| {row['model']} | {row['split']} | {row['status']} | {row['samples']} "
            f"| {row['S'] if row['S'] is not None else '-'} "
            f"| {row['D'] if row['D'] is not None else '-'} "
            f"| {row['I'] if row['I'] is not None else '-'} "
            f"| {row['N'] if row['N'] is not None else '-'} "
            f"| {percent(row['CER'])} | {percent(row['WA'])} | {percent(row['CA'])} "
            f"| {runtime} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, markdown_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL-7B, GOT-OCR2.0 and PaddleOCR-VL via API."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys from the YAML; default: all configured models.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["validation", "val", "test"],
        default=None,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Deterministic pilot cap per split. Omit for formal full-split evaluation.",
    )
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-errors", type=int, default=None)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare/check manifests without checking API credentials or calling APIs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this run-tag's generated cache/results before calling APIs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    config_path = _project_path(args.config)
    config = load_yaml(config_path)
    paths_config_path = _project_path(config["experiment"]["paths_config"])
    paths_config = load_yaml(paths_config_path)
    output_root = _resolve_output_root(paths_config, args.output_root)
    all_models = dict(config["models"])
    selected_model_keys = list(args.models or all_models.keys())
    unknown = sorted(set(selected_model_keys).difference(all_models))
    if unknown:
        raise ValueError(
            f"Unknown model key(s): {unknown}. Available: {sorted(all_models)}"
        )
    selected_splits = [
        normalize_split_name(value)
        for value in (args.splits or config["protocol"].get("default_splits", ["validation", "test"]))
    ]
    selected_splits = sorted(
        set(selected_splits),
        key=lambda value: {"validation": 0, "test": 1}[value],
    )
    run_tag = args.run_tag or (
        "full_v1" if args.max_samples is None else f"pilot_n{args.max_samples}"
    )
    _slug(run_tag)

    prepared = prepare_splits(
        experiment_config=config,
        paths_config=paths_config,
        selected_splits=selected_splits,
        output_root=output_root,
        run_tag=run_tag,
        max_samples=args.max_samples,
        skip_image_check=args.skip_image_check,
        overwrite=args.overwrite,
    )
    total_calls = len(selected_model_keys) * sum(
        len(prepared[split].selected_samples) for split in selected_splits
    )
    print(f"[PLAN] config={config_path}")
    print(f"[PLAN] paths_config={paths_config_path}")
    print(f"[PLAN] output_root={output_root}")
    print(f"[PLAN] run_tag={run_tag} zero_shot=true training=false")
    print(f"[PLAN] models={','.join(selected_model_keys)}")
    print(f"[PLAN] splits={','.join(selected_splits)}")
    for split in selected_splits:
        item = prepared[split]
        print(
            f"[PLAN] split={split} source_samples={len(item.source_samples)} "
            f"selected_samples={len(item.selected_samples)} full_split={str(item.full_split).lower()} "
            f"manifest={item.evaluation_manifest}"
        )
    print(f"[PLAN] maximum_api_calls={total_calls}")

    if args.prepare_only:
        print("[OK] Manifests prepared; API credentials were not checked and no request was sent.")
        return 0

    issues: list[str] = []
    for model_key in selected_model_keys:
        model_issues = validate_provider_environment(all_models[model_key])
        issues.extend(f"{model_key}: {issue}" for issue in model_issues)
    if issues:
        for issue in issues:
            print(f"[MISSING] {issue}", file=sys.stderr)
        print(
            "[ERROR] API preflight failed; no API request was sent. "
            "See docs/lmm_zero_shot_api_experiment.md.",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        print("[OK] Data, dependencies and credential environment variables passed preflight; no API request was sent.")
        return 0

    outcomes: list[RunOutcome] = []
    # Split-major order ensures every validation run is completed before the
    # held-out test is touched when both are requested together.
    for split in selected_splits:
        split_outcomes: list[RunOutcome] = []
        for model_key in selected_model_keys:
            outcome = run_model_split(
                experiment_config=config,
                config_path=config_path,
                output_root=output_root,
                run_tag=run_tag,
                model_key=model_key,
                model_config=all_models[model_key],
                prepared_split=prepared[split],
                workers_override=args.workers,
                max_errors_override=args.max_errors,
                overwrite=args.overwrite,
            )
            outcomes.append(outcome)
            split_outcomes.append(outcome)
        if (
            split == "validation"
            and "test" in selected_splits
            and any(item.status != "completed" for item in split_outcomes)
        ):
            print(
                "[STOP] Validation is incomplete; held-out test was not touched. "
                "Fix the API issue and rerun the same command.",
                file=sys.stderr,
            )
            break

    csv_path, json_path, markdown_path = write_summary(
        outcomes=outcomes,
        output_root=output_root,
        protocol_version=str(config["experiment"]["protocol_version"]),
        run_tag=run_tag,
    )
    failures = sum(outcome.status != "completed" for outcome in outcomes)
    print(f"[SUMMARY] csv={csv_path}")
    print(f"[SUMMARY] json={json_path}")
    print(f"[SUMMARY] markdown={markdown_path}")
    print(f"[SUMMARY] completed={len(outcomes) - failures} incomplete={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
