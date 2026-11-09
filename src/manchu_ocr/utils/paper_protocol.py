"""Prevent accidentally continuing a paper experiment from legacy objectives."""
from copy import deepcopy
import math
import json
from pathlib import Path

from manchu_ocr.data.orthography import sha256_file, HARMONY_PAIRS, SPELLING_PAIRS

PAPER_PROTOCOL = "paper_v2_20261109"


def bind_transition_prior(cfg, recognition_data):
    if not cfg.get("loss", {}).get("use_orthographic_loss"):
        return
    path = Path(cfg["loss"]["transition_matrix"])
    digest = sha256_file(path)
    if cfg.get("experiment", {}).get("protocol_version") == PAPER_PROTOCOL:
        meta_path = path.with_suffix(".json")
        if not meta_path.is_file():
            raise ValueError("Paper OTP requires a rebuilt prior with provenance JSON; run tools/build_transition_matrix.py")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {"schema": "manchu.transition_prior.v2", "source_split": "train",
                    "rho": cfg["experiment"].get("rho", 0.05), "smoothing": 1.0, "epsilon_t": 1e-6,
                    "train_manifest_sha256": sha256_file(recognition_data["train_list"]),
                    "charset_sha256": sha256_file(recognition_data["charset"]), "matrix_sha256": digest,
                    "harmony_pairs": [list(p) for p in sorted(HARMONY_PAIRS)],
                    "spelling_pairs": [list(p) for p in sorted(SPELLING_PAIRS)]}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"Transition prior provenance mismatch: {key}")
    cfg["experiment"]["transition_matrix_sha256"] = digest


def is_better_checkpoint(metric, cer, best_metric, best_cer):
    if not math.isfinite(metric) or not math.isfinite(cer):
        raise ValueError("Checkpoint selection requires finite validation metrics")
    return metric > best_metric or (metric == best_metric and cer < best_cer)


def validate_paper_checkpoint(cfg, checkpoint, *, for_resume=False):
    if cfg.get("experiment", {}).get("protocol_version") != PAPER_PROTOCOL:
        return
    old = checkpoint.get("config", {})
    if old.get("experiment", {}).get("protocol_version") != PAPER_PROTOCOL:
        raise ValueError("Legacy checkpoint cannot be used for the new paper protocol; retrain from epoch 1")
    for section in ("model", "loss", "decode"):
        current, previous = deepcopy(cfg.get(section, {})), deepcopy(old.get(section, {}))
        if section == "loss":
            current.pop("transition_matrix", None)
            previous.pop("transition_matrix", None)
        if current != previous:
            raise ValueError(f"Paper checkpoint {section} differs from the requested configuration")
    if for_resume and cfg.get("data") != old.get("data"):
        raise ValueError("Resume data/augmentation configuration differs")
    expected = cfg.get("experiment", {}).get("transition_matrix_sha256")
    if expected and old.get("experiment", {}).get("transition_matrix_sha256") != expected:
        raise ValueError("Transition prior changed since training; start a new experiment")
