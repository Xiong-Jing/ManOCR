import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

from manchu_ocr.models.detection.builder import build_detection_model
from manchu_ocr.models.recognition.builder import build_recognition_model
from manchu_ocr.utils.config import load_yaml


DETECTION_CONFIGS = [
    "configs/detection/dbnetpp_official_baseline.yaml",
    "configs/detection/dbnetpp_vsaa.yaml",
    "configs/detection/dbnetpp_asym_shrink.yaml",
    "configs/detection/dbnetpp_vsaa_asym_shrink.yaml",
]

RECOGNITION_CONFIGS = [
    "configs/recognition/svtr_official_baseline.yaml",
    "configs/recognition/svtr_official_dab.yaml",
    "configs/recognition/svtr_official_lortho.yaml",
    "configs/recognition/svtr_official_dab_lortho.yaml",
]

EXPERIMENT_CONFIGS = [
    "configs/experiments/det_ablation.yaml",
    "configs/experiments/rec_ablation.yaml",
    "configs/experiments/det_main.yaml",
    "configs/experiments/rec_main.yaml",
    "configs/experiments/full_pipeline.yaml",
]

RUN_SCRIPTS = [
    "scripts/run_det_ablation.sh",
    "scripts/run_det_degraded_ablation.sh",
    "scripts/run_rec_ablation.sh",
    "scripts/run_full_pipeline.sh",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def fail(self, message: str) -> None:
        print(f"[FAIL] {message}")
        self.errors.append(message)

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.ok(message)
        else:
            self.fail(message)

    def finish(self) -> None:
        if self.errors:
            print("")
            print(f"[RESULT] preflight failed: {len(self.errors)} issue(s)")
            for idx, error in enumerate(self.errors, start=1):
                print(f"{idx}. {error}")
            raise SystemExit(1)

        print("")
        print("[RESULT] preflight passed")


def count_files(path: Path, suffixes: Iterable[str] | None = None) -> int:
    if not path.exists():
        return 0

    if suffixes is None:
        return sum(1 for item in path.iterdir() if item.is_file())

    suffixes = {suffix.lower() for suffix in suffixes}
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def check_yaml_parse(audit: Audit) -> None:
    for path in Path("configs").rglob("*.yaml"):
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            audit.fail(f"Invalid YAML: {path}: {exc}")
            continue
    audit.ok("all YAML files parse")


def check_scripts(audit: Audit) -> None:
    for script in RUN_SCRIPTS:
        path = Path(script)
        audit.require(path.exists(), f"run script exists: {script}")
        if path.exists():
            data = path.read_bytes()
            audit.require(b"\r\n" not in data, f"run script uses LF line endings: {script}")


def check_experiment_references(audit: Audit) -> None:
    for path in EXPERIMENT_CONFIGS:
        cfg_path = Path(path)
        if not cfg_path.exists():
            audit.fail(f"Missing experiment config: {path}")
            continue

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        for exp in cfg.get("experiments", []):
            target = Path(exp["config"])
            audit.require(target.exists(), f"{path} references existing config: {target}")

        for model in cfg.get("models", []):
            if "config" in model:
                target = Path(model["config"])
                audit.require(target.exists(), f"{path} references existing model config: {target}")

        workflow = cfg.get("workflow")
        if isinstance(workflow, dict):
            for key in ["script", "train_script", "eval_script", "summary_script", "plot_script"]:
                if key in workflow:
                    target = Path(workflow[key])
                    audit.require(target.exists(), f"{path} references existing workflow {key}: {target}")

        for section in ["main_model", "detection", "recognition"]:
            item = cfg.get(section)
            if isinstance(item, dict) and "config" in item:
                target = Path(item["config"])
                audit.require(target.exists(), f"{path} references existing config: {target}")


def check_raw_data(audit: Audit, paths_cfg: dict) -> None:
    det = paths_cfg["detection_data"]
    rec = paths_cfg["recognition_data"]

    for name, value in [
        ("project_root", paths_cfg.get("project_root")),
        ("detection root", det.get("root")),
        ("detection raw_images", det.get("raw_images")),
        ("detection raw_annotations", det.get("raw_annotations")),
        ("recognition root", rec.get("root")),
        ("recognition raw_images", rec.get("raw_images")),
        ("recognition raw_excel", rec.get("raw_excel")),
    ]:
        if value is None:
            audit.fail(f"Missing path key: {name}")
            continue
        audit.require(Path(value).exists(), f"{name} exists: {value}")

    raw_images = Path(det["raw_images"])
    raw_annotations = Path(det["raw_annotations"])
    image_count = count_files(raw_images, IMAGE_EXTS)
    json_count = count_files(raw_annotations, {".json"})
    audit.info(f"detection raw image count: {image_count}")
    audit.info(f"detection raw JSON count: {json_count}")

    expected_images = det.get("expected_raw_images")
    expected_json = det.get("expected_raw_annotations")

    if expected_images is not None:
        audit.require(image_count >= int(expected_images), f"detection raw images >= {expected_images}")
    if expected_json is not None:
        audit.require(json_count >= int(expected_json), f"detection raw JSON >= {expected_json}")

    rec_image_count = 0
    rec_raw_images = Path(rec["raw_images"])
    if rec_raw_images.exists():
        rec_image_count = sum(
            1 for path in rec_raw_images.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
    audit.info(f"recognition raw image count: {rec_image_count}")


def read_manifest_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def check_processed_data(audit: Audit, paths_cfg: dict) -> None:
    det = paths_cfg["detection_data"]
    rec = paths_cfg["recognition_data"]

    for name, value in [
        ("detection clean_annotations", det.get("clean_annotations")),
        ("detection train_list", det.get("train_list")),
        ("detection val_list", det.get("val_list")),
        ("detection test_list", det.get("test_list")),
        ("recognition labels_csv", rec.get("labels_csv")),
        ("recognition charset", rec.get("charset")),
        ("recognition transition_matrix", rec.get("transition_matrix")),
        ("recognition train_list", rec.get("train_list")),
        ("recognition val_list", rec.get("val_list")),
        ("recognition test_list", rec.get("test_list")),
    ]:
        if value is None:
            audit.fail(f"Missing processed path key: {name}")
            continue
        audit.require(Path(value).exists(), f"{name} exists: {value}")

    for name, value in [
        ("detection train", det.get("train_list")),
        ("detection val", det.get("val_list")),
        ("detection test", det.get("test_list")),
        ("recognition train", rec.get("train_list")),
        ("recognition val", rec.get("val_list")),
        ("recognition test", rec.get("test_list")),
    ]:
        if value is None:
            continue
        count = read_manifest_count(Path(value))
        audit.info(f"{name} manifest samples: {count}")
        audit.require(count > 0, f"{name} manifest is non-empty")

    charset_path = Path(rec["charset"])
    matrix_path = Path(rec["transition_matrix"])

    if charset_path.exists() and matrix_path.exists():
        charset = [line for line in charset_path.read_text(encoding="utf-8").splitlines() if line]
        matrix = np.load(matrix_path)
        audit.info(f"recognition charset size: {len(charset)}")
        audit.info(f"transition matrix shape: {matrix.shape}")
        audit.require(matrix.shape == (len(charset), len(charset)), "transition matrix matches charset size")


def check_config_values(audit: Audit) -> None:
    seen_exp_names = set()

    for cfg_path in DETECTION_CONFIGS + RECOGNITION_CONFIGS:
        cfg = load_yaml(cfg_path)
        exp_name = cfg["experiment"]["name"]
        audit.require(exp_name not in seen_exp_names, f"unique experiment name: {exp_name}")
        seen_exp_names.add(exp_name)
        audit.require(
            cfg["experiment"]["paths_config"] == "configs/paths/remote_server.yaml",
            f"{cfg_path} uses remote_server paths_config",
        )

    for cfg_path in DETECTION_CONFIGS:
        cfg = load_yaml(cfg_path)
        audit.require(int(cfg["data"]["batch_size"]) >= 1, f"{cfg_path} detection batch_size valid")
        audit.require("eval" in cfg, f"{cfg_path} has eval postprocess config")
        audit.require(
            "augmentation" in cfg["data"],
            f"{cfg_path} has detection train augmentation config",
        )
        audit.require(
            not bool(cfg["data"].get("augmentation", {}).get("enabled", True)),
            f"{cfg_path} detection train augmentation disabled",
        )
        audit.require(
            bool(cfg.get("eval_degradation", {}).get("enabled", False)),
            f"{cfg_path} degraded validation/test enabled",
        )

    for cfg_path in RECOGNITION_CONFIGS:
        cfg = load_yaml(cfg_path)
        audit.require(int(cfg["data"]["batch_size"]) >= 1, f"{cfg_path} recognition batch_size valid")
        if cfg["loss"].get("use_orthographic_loss", False):
            audit.require("transition_matrix" in cfg["loss"], f"{cfg_path} has transition_matrix for Lortho")


def check_model_builds(audit: Audit) -> None:
    for cfg_path in DETECTION_CONFIGS:
        try:
            cfg = load_yaml(cfg_path)
            model = build_detection_model(cfg).eval()
            with torch.no_grad():
                preds = model(torch.randn(1, 3, 128, 96))
            ok_shape = tuple(preds["prob_map"].shape) == (1, 1, 128, 96)
            audit.require(ok_shape, f"detection model forward shape ok: {cfg_path}")
        except Exception as exc:
            audit.fail(f"detection model build/forward failed: {cfg_path}: {exc}")

    for cfg_path in RECOGNITION_CONFIGS:
        try:
            cfg = load_yaml(cfg_path)
            model = build_recognition_model(cfg, num_classes=128).eval()
            height = int(cfg["data"]["image_height"])
            width = int(cfg["data"]["image_width"])
            with torch.no_grad():
                logits = model(torch.randn(1, 3, height, width))
            ok_shape = logits.ndim == 3 and logits.shape[0] == 1 and logits.shape[-1] == 128
            audit.require(ok_shape, f"recognition model forward shape ok: {cfg_path}")
        except Exception as exc:
            audit.fail(f"recognition model build/forward failed: {cfg_path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paths/remote_server.yaml")
    parser.add_argument(
        "--stage",
        choices=["raw", "processed", "all"],
        default="all",
        help="Use raw after transfer, processed/all before formal training.",
    )
    parser.add_argument("--skip-model-build", action="store_true")
    args = parser.parse_args()

    audit = Audit()

    check_yaml_parse(audit)
    check_scripts(audit)
    check_experiment_references(audit)

    paths_cfg = load_yaml(args.config)
    check_config_values(audit)

    if args.stage in {"raw", "all"}:
        check_raw_data(audit, paths_cfg)

    if args.stage in {"processed", "all"}:
        check_processed_data(audit, paths_cfg)

    if not args.skip_model_build:
        check_model_builds(audit)

    audit.finish()


if __name__ == "__main__":
    main()
