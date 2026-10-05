import argparse
import json
from collections import Counter
from pathlib import Path

from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import save_json
from manchu_ocr.utils.logger import setup_logger


def summarize_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = {
        "file": str(path).replace("\\", "/"),
        "top_level_type": type(data).__name__,
        "top_level_keys": [],
        "num_shapes": None,
        "shape_keys": [],
        "shape_types": [],
        "example_shape": None,
    }

    if isinstance(data, dict):
        summary["top_level_keys"] = list(data.keys())

        if "shapes" in data and isinstance(data["shapes"], list):
            shapes = data["shapes"]
            summary["num_shapes"] = len(shapes)

            if shapes:
                shape_key_counter = Counter()
                shape_type_counter = Counter()

                for shape in shapes:
                    if isinstance(shape, dict):
                        shape_key_counter.update(shape.keys())
                        if "shape_type" in shape:
                            shape_type_counter.update([shape.get("shape_type")])

                summary["shape_keys"] = list(shape_key_counter.keys())
                summary["shape_types"] = list(shape_type_counter.keys())
                summary["example_shape"] = shapes[0]

    elif isinstance(data, list):
        summary["num_shapes"] = len(data)
        if data:
            summary["example_shape"] = data[0]
            if isinstance(data[0], dict):
                summary["shape_keys"] = list(data[0].keys())

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paths/remote_server.yaml",
    )
    parser.add_argument("--max-files", type=int, default=10)
    args = parser.parse_args()

    logger = setup_logger("inspect_detection_json")
    cfg = load_yaml(args.config)

    ann_dir = Path(cfg["detection_data"]["raw_annotations"])
    out_path = Path(cfg["detection_data"]["root"]) / "processed" / "detection_json_schema_summary.json"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    json_files = sorted(ann_dir.glob("*.json"))

    if not json_files:
        raise RuntimeError(f"No JSON files found in: {ann_dir}")

    summaries = []

    for path in json_files[: args.max_files]:
        summary = summarize_json_file(path)
        summaries.append(summary)

        logger.info(f"File: {path.name}")
        logger.info(f"Top keys: {summary['top_level_keys']}")
        logger.info(f"Num shapes: {summary['num_shapes']}")
        logger.info(f"Shape keys: {summary['shape_keys']}")
        logger.info(f"Shape types: {summary['shape_types']}")

    save_json(
        {
            "annotation_dir": str(ann_dir).replace("\\", "/"),
            "num_json_files": len(json_files),
            "checked_files": len(summaries),
            "summaries": summaries,
        },
        out_path,
    )

    logger.info(f"Saved schema summary to: {out_path}")


if __name__ == "__main__":
    main()

