import argparse
from pathlib import Path

from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.logger import setup_logger


def check_path(name: str, path: str | Path, should_exist: bool = True, create: bool = False) -> bool:
    path = Path(path)

    if create:
        path.mkdir(parents=True, exist_ok=True)

    exists = path.exists()

    if should_exist and exists:
        print(f"[OK] {name}: {path}")
        return True

    if should_exist and not exists:
        print(f"[MISSING] {name}: {path}")
        return False

    print(f"[INFO] {name}: {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paths/remote_server.yaml",
        help="Path to paths yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    logger = setup_logger("check_paths")
    logger.info(f"Loaded config: {args.config}")

    all_ok = True

    # project root
    if "project_root" in cfg:
        all_ok &= check_path("project_root", cfg["project_root"])

    # detection data
    det = cfg.get("detection_data", {})
    if det:
        all_ok &= check_path("detection_data.root", det.get("root", ""))
        all_ok &= check_path("detection_data.raw_images", det.get("raw_images", ""))
        all_ok &= check_path("detection_data.raw_annotations", det.get("raw_annotations", ""))

        # processed paths may not exist yet, so create them if needed
        if det.get("clean_annotations"):
            all_ok &= check_path(
                "detection_data.clean_annotations",
                det["clean_annotations"],
                should_exist=True,
                create=True,
            )

    # recognition data
    rec = cfg.get("recognition_data", {})
    if rec:
        all_ok &= check_path("recognition_data.root", rec.get("root", ""))
        all_ok &= check_path("recognition_data.raw_images", rec.get("raw_images", ""))

        if rec.get("raw_excel"):
            all_ok &= check_path("recognition_data.raw_excel", rec["raw_excel"])

    # outputs
    outputs = cfg.get("outputs", {})
    if outputs:
        for key, value in outputs.items():
            all_ok &= check_path(f"outputs.{key}", value, should_exist=True, create=True)

    if all_ok:
        logger.info("All required paths are ready.")
    else:
        logger.error("Some required paths are missing. Please fix them before continuing.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()


