"""Count actual model parameters; optionally benchmark model-only forward passes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.experiment_result import collect_torch_runtime
from manchu_ocr.data.orthography import sha256_file


def parameter_counts(model):
    # nn.Module.parameters() deduplicates shared parameters; buffers/EMA/optimizer excluded.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": total, "trainable_parameters": trainable, "parameters_M": total / 1e6}


@torch.inference_mode()
def benchmark(model, shape, device, warmup=10, iterations=50):
    if warmup < 0 or iterations <= 0 or any(v <= 0 for v in shape):
        raise ValueError("Invalid benchmark budget/input shape")
    model.to(device).eval()
    inputs = torch.zeros(shape, device=device)
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    for _ in range(warmup):
        model(inputs)
    sync()
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        model(inputs)
        sync()
        times.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {"scope": "model forward only; excludes image IO, preprocessing, decoding/postprocessing",
            "input_shape": list(shape), "dtype": "float32", "warmup": warmup, "iterations": iterations,
            "mean_batch_ms": float(np.mean(times) * 1000), "median_batch_ms": float(np.median(times) * 1000),
            "p95_batch_ms": float(np.percentile(times, 95) * 1000),
            "samples_per_second": shape[0] / float(np.mean(times)),
            "peak_allocated_bytes": peak, "forward_extra_allocated_bytes": None if peak is None else peak - baseline}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Optional; validate/load the actual trained model")
    parser.add_argument("--charset", help="Required for recognition unless paths config supplies it")
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    task = cfg["experiment"]["task"]
    build_cfg = copy.deepcopy(cfg)
    state = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "config" in state:
            build_cfg["model"] = state["config"]["model"]
    if task == "detection":
        from manchu_ocr.models.detection.builder import build_detection_model
        if isinstance(build_cfg["model"].get("backbone"), dict):
            build_cfg["model"]["backbone"]["pretrained"] = False
        model = build_detection_model(build_cfg)
    elif task == "recognition":
        from manchu_ocr.models.recognition.builder import build_recognition_model
        from manchu_ocr.data.label_converters.ctc_label_converter import CTCLabelConverter
        charset = args.charset or load_yaml(cfg["experiment"]["paths_config"])["recognition_data"]["charset"]
        model = build_recognition_model(build_cfg, CTCLabelConverter(charset).num_classes)
    else:
        raise ValueError(f"Unsupported task: {task}")
    if state is not None:
        model.load_state_dict(state["model"], strict=True)
    report = {"config": args.config, "config_sha256": sha256_file(args.config),
              "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
              "model_config": build_cfg["model"], **parameter_counts(model)}
    if args.benchmark:
        data = cfg["data"]
        height = data["target_height" if task == "detection" else "image_height"]
        width = data["target_width" if task == "detection" else "image_width"]
        shape = (args.batch_size, build_cfg["model"].get("in_channels", 3), height, width)
        report["benchmark"] = benchmark(model, shape, torch.device(args.device), args.warmup, args.iterations)
    report["runtime"] = collect_torch_runtime(args.device if args.benchmark else "cpu")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
