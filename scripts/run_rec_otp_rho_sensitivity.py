"""Prepare and run the controlled rho sweep; no paper scores are embedded."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from manchu_ocr.data.orthography import (RULE_PAIRS, build_transition_matrix,
                                       sha256_file, write_transition_artifact)
from manchu_ocr.utils.config import load_yaml, save_yaml

ROOT = Path(__file__).resolve().parents[1]


def rho_name(rho):
    return "svtr_dab_otp_rho_" + format(float(rho), "g").replace(".", "p")


def make_config(base, rho, matrix_path, paths_path, protocol):
    cfg = copy.deepcopy(base)
    cfg["experiment"].update(name=rho_name(rho), paths_config=str(paths_path))
    cfg["loss"].update(use_orthographic_loss=True, lambda_ortho=0.1,
                       use_alignment_loss=False, lambda_align=0.0, lambda_align_final=0.0,
                       normalize_char_probs=True, confidence_weighting=True,
                       confidence_power=1.0, min_confidence=0.0, transition_matrix=str(matrix_path))
    # Old scheduling keys must never change the fixed loss coefficient.
    for key in list(cfg["loss"]):
        if key.startswith("lambda_ortho_"):
            del cfg["loss"][key]
    cfg["decode"].update(use_lexicon=False, use_ctc_lexicon_rerank=False)
    cfg["train"].update(epochs=protocol["epochs"], seed=protocol["seed"], best_metric="word_accuracy")
    cfg["experiment"]["rho"] = float(rho)
    return cfg


def write_frozen_yaml(cfg, path):
    if path.exists():
        if load_yaml(path) != cfg:
            raise ValueError(f"Frozen configuration changed: {path}; use a new output directory")
    else:
        save_yaml(cfg, path)


def prepare(plan, paths, base, root):
    protocol = plan["protocol"]
    rec = paths["recognition_data"]
    manifest, charset_path = Path(rec["train_list"]), Path(rec["charset"])
    labels = []
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            parts = line.split("\t", 1)
            if len(parts) != 2 or not parts[1]:
                raise ValueError("Malformed training manifest")
            labels.append(parts[1])
    if not labels:
        raise ValueError("Empty training manifest")
    charset = charset_path.read_text(encoding="utf-8-sig").splitlines()
    root.mkdir(parents=True, exist_ok=True)
    paths_path = root / "paths.yaml"
    write_frozen_yaml(paths, paths_path)
    configs = []
    for rho in plan["rhos"]:
        name = rho_name(rho)
        matrix_path = root / f"{name}.npy"
        matrix = build_transition_matrix(labels, charset, protocol["smoothing"], rho, protocol["epsilon_t"])
        if matrix_path.exists():
            meta = json.loads(matrix_path.with_suffix(".json").read_text(encoding="utf-8"))
            if (not np.array_equal(np.load(matrix_path), matrix)
                    or meta["train_manifest_sha256"] != sha256_file(manifest)
                    or meta["charset_sha256"] != sha256_file(charset_path)
                    or meta["matrix_sha256"] != sha256_file(matrix_path)):
                raise ValueError(f"Frozen prior/source changed: {matrix_path}")
        else:
            write_transition_artifact(matrix_path, matrix, manifest=manifest, charset_path=charset_path,
                                      rho=rho, smoothing=protocol["smoothing"], epsilon_t=protocol["epsilon_t"])
        cfg = make_config(base, rho, matrix_path, paths_path, protocol)
        config_path = root / f"{name}.yaml"
        write_frozen_yaml(cfg, config_path)
        configs.append((rho, config_path))
    return configs


def summarize(configs, output_root, splits):
    rows = []
    for rho, cfg_path in configs:
        cfg = load_yaml(cfg_path)
        for split in splits:
            path = output_root / "metrics/recognition" / rho_name(rho) / f"eval_{split}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            m = result["metrics"]
            expected_split = "validation" if split == "val" else split
            if result.get("split") != expected_split or int(m.get("num_samples", 0)) <= 0:
                raise ValueError(f"Invalid or empty evaluation: {path}")
            checkpoint = output_root / "checkpoints/recognition" / rho_name(rho) / "best.pth"
            if Path(result["checkpoint_path"]).resolve() != checkpoint.resolve():
                raise ValueError(f"Wrong checkpoint: {path}")
            paths = load_yaml(cfg["experiment"]["paths_config"])
            expected_hashes = {
                "checkpoint_sha256": sha256_file(checkpoint),
                "config_sha256": sha256_file(cfg_path),
                "manifest_sha256": sha256_file(paths["recognition_data"][f"{split}_list"]),
                "transition_matrix_sha256": sha256_file(cfg["loss"]["transition_matrix"]),
            }
            if result.get("rho") != rho or any(result.get(k) != v for k, v in expected_hashes.items()):
                raise ValueError(f"Stale or mismatched rho evaluation: {path}")
            rows.append({"rho": rho, "lambda_ortho": 0.1, "split": expected_split,
                         "WA": m["word_accuracy"], "CA": m["character_accuracy"], "CER": m["cer"],
                         "prior_sha256": sha256_file(cfg["loss"]["transition_matrix"]),
                         "checkpoint_sha256": sha256_file(checkpoint), "result": str(path)})
    dest = output_root / "metrics/recognition/recognition_otp_rho_sensitivity"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with dest.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/experiments/rec_otp_rho_sensitivity.yaml")
    parser.add_argument("--paths-config")
    parser.add_argument("--output-root")
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate", "summarize", "all"], default="prepare")
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = load_yaml(args.plan)
    if float(plan["protocol"]["lambda_ortho"]) != 0.1 or set(map(tuple, plan["protocol"]["rule_pairs"])) != RULE_PAIRS:
        raise ValueError("rho experiment must fix lambda=0.1 and the declared paper rule pairs")
    rhos = list(map(float, plan["rhos"]))
    if not rhos or len(set(rhos)) != len(rhos) or any(not 0 < r <= 1 for r in rhos):
        raise ValueError("Invalid rho sweep")
    base = load_yaml(plan["base_config"])
    paths = load_yaml(args.paths_config or base["experiment"]["paths_config"])
    output = Path(args.output_root or paths["outputs"]["root"]).resolve()
    paths["outputs"]["root"] = str(output)
    run_root = output / "experiments/otp_rho"
    configs = [(rho, run_root / f"{rho_name(rho)}.yaml") for rho in rhos]
    commands = []
    for rho, config in configs:
        checkpoint = output / "checkpoints/recognition" / rho_name(rho) / "best.pth"
        if args.stage in {"train", "all"}:
            command = [sys.executable, "scripts/train_recognition.py", "--config", str(config)]
            if args.resume:
                command += ["--resume", str(checkpoint.with_name("last.pth"))]
            elif not args.dry_run and checkpoint.with_name("last.pth").exists():
                raise ValueError("Existing run: use --resume or a new --output-root")
            commands.append(command)
        if args.stage in {"evaluate", "all"}:
            for split in args.splits:
                commands.append([sys.executable, "scripts/eval_recognition.py", "--config", str(config),
                                 "--checkpoint", str(checkpoint), "--split", split, "--save-predictions"])
    if args.dry_run:
        print(json.dumps({"prepare_priors": rhos, "fixed_lambda": 0.1, "commands": commands}, indent=2))
        return 0
    configs = prepare(plan, paths, base, run_root)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.stage in {"evaluate", "all", "summarize"}:
        summarize(configs, output, args.splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
