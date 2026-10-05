import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from manchu_ocr.data.label_generators.polygon_utils import (
    bbox_iou,
    clip_polygon_to_image,
    is_valid_polygon,
    normalize_points,
    polygon_area,
    polygon_bbox,
    rectangle_to_polygon,
)
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import ensure_dir, save_json, write_txt
from manchu_ocr.utils.logger import setup_logger
from manchu_ocr.utils.seed import set_seed


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_matching_image(json_path: Path, image_dir: Path) -> Path | None:
    stem = json_path.stem

    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # 兼容大小写后缀
    for path in image_dir.iterdir():
        if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTS:
            return path

    return None


def get_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def extract_polygons_from_labelme(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    shapes = data.get("shapes", [])

    polygons = []

    for idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue

        points = shape.get("points")
        if not points:
            continue

        shape_type = shape.get("shape_type", "polygon")

        try:
            if shape_type == "rectangle":
                polygon = rectangle_to_polygon(points)
            else:
                polygon = normalize_points(points)
        except Exception:
            continue

        polygons.append(
            {
                "id": idx,
                "label": shape.get("label", "text"),
                "points": polygon,
                "shape_type": shape_type,
                "original": shape,
            }
        )

    return polygons


def extract_polygons_from_list(data: List[Any]) -> List[Dict[str, Any]]:
    polygons = []

    for idx, item in enumerate(data):
        points = None
        label = "text"

        if isinstance(item, dict):
            if "points" in item:
                points = item["points"]
            elif "polygon" in item:
                points = item["polygon"]
            elif "bbox" in item:
                box = item["bbox"]
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    points = [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ]

            label = item.get("label", item.get("text", "text"))
        elif isinstance(item, list):
            points = item

        if not points:
            continue

        try:
            polygon = normalize_points(points)
        except Exception:
            continue

        polygons.append(
            {
                "id": idx,
                "label": label,
                "points": polygon,
                "shape_type": "polygon",
                "original": item,
            }
        )

    return polygons


def extract_polygons(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "shapes" in data:
        return extract_polygons_from_labelme(data)

    if isinstance(data, list):
        return extract_polygons_from_list(data)

    return []


def clean_polygons(
    polygons: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    max_area_ratio: float,
    duplicate_iou_threshold: float,
    min_area: float,
    remove_side_band: bool = False,
    side_band_ratio: float = 0.10,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "raw_polygons": len(polygons),
        "invalid_polygons": 0,
        "large_outer_boxes": 0,
        "side_band_boxes": 0,
        "duplicate_boxes": 0,
        "kept_polygons": 0,
    }

    image_area = float(image_width * image_height)

    candidates = []

    for poly in polygons:
        points = clip_polygon_to_image(poly["points"], image_width, image_height)

        if not is_valid_polygon(points, min_area=min_area):
            stats["invalid_polygons"] += 1
            continue

        area = polygon_area(points)
        area_ratio = area / max(image_area, 1.0)

        if area_ratio > max_area_ratio:
            stats["large_outer_boxes"] += 1
            continue

        x1, y1, x2, y2 = polygon_bbox(points)

        # 删除左右侧栏异常框，例如黑色书签、侧边装饰、页边部件
        if remove_side_band:
            cx = (x1 + x2) / 2.0
            left_boundary = image_width * side_band_ratio
            right_boundary = image_width * (1.0 - side_band_ratio)

            if cx < left_boundary or cx > right_boundary:
                stats["side_band_boxes"] += 1
                continue

        new_poly = {
            "id": poly["id"],
            "label": poly.get("label", "text"),
            "points": points,
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "area_ratio": area_ratio,
            "shape_type": poly.get("shape_type", "polygon"),
        }

        candidates.append(new_poly)

    # 去重：面积大的优先保留；如果 IoU 极高，认为重复
    candidates = sorted(candidates, key=lambda x: x["area"], reverse=True)

    kept = []

    for poly in candidates:
        box = tuple(poly["bbox"])
        duplicated = False

        for kept_poly in kept:
            kept_box = tuple(kept_poly["bbox"])
            iou = bbox_iou(box, kept_box)

            if iou >= duplicate_iou_threshold:
                duplicated = True
                stats["duplicate_boxes"] += 1
                break

        if not duplicated:
            kept.append(poly)

    # 恢复按页面位置排序：先 x，再 y，适合左到右列、列内自上而下的初步顺序
    kept = sorted(kept, key=lambda x: (x["bbox"][0], x["bbox"][1]))

    stats["kept_polygons"] = len(kept)

    return kept, stats


def save_clean_annotation(
    output_path: Path,
    image_path: Path,
    image_width: int,
    image_height: int,
    polygons: List[Dict[str, Any]],
    source_json: Path,
) -> None:
    data = {
        "image_path": str(image_path).replace("\\", "/"),
        "image_width": image_width,
        "image_height": image_height,
        "source_json": str(source_json).replace("\\", "/"),
        "polygons": [
            {
                "label": poly["label"],
                "points": [[float(x), float(y)] for x, y in poly["points"]],
                "bbox": [float(v) for v in poly["bbox"]],
                "area": float(poly["area"]),
                "area_ratio": float(poly["area_ratio"]),
            }
            for poly in polygons
        ],
    }

    save_json(data, output_path)


def split_items(
    items: List[Tuple[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: Optional[float] = None,
):
    ratio_sum = train_ratio + val_ratio + (test_ratio if test_ratio is not None else 0.0)

    if test_ratio is not None and abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios should sum to 1.0, got "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}, sum={ratio_sum}"
        )

    if test_ratio is None and train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio should be < 1.0 when test_ratio is omitted, "
            f"got train={train_ratio}, val={val_ratio}"
        )

    items = items[:]
    random.Random(seed).shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    return train, val, test


def save_manifest(items: List[Tuple[str, str]], path: Path) -> None:
    lines = [f"{img}\t{ann}" for img, ann in items]
    write_txt(lines, path)


def remove_stale_or_write(data, path: Path, writer) -> None:
    if data:
        writer(data, path)
    elif path.exists():
        path.unlink()


def clear_clean_annotation_dir(clean_ann_dir: Path, data_root: Path, logger) -> None:
    if not clean_ann_dir.exists():
        return

    resolved_clean = clean_ann_dir.resolve()
    resolved_root = data_root.resolve()

    if not str(resolved_clean).lower().startswith(str(resolved_root).lower()):
        raise RuntimeError(
            f"Refusing to clear clean annotation dir outside detection root: {resolved_clean}"
        )

    removed = 0

    for path in clean_ann_dir.glob("*.json"):
        path.unlink()
        removed += 1

    logger.info(f"Cleared old clean annotation JSON files: {removed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paths/remote_server.yaml",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--max-area-ratio", type=float, default=None)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=None)
    parser.add_argument("--min-area", type=float, default=None)
    parser.add_argument(
        "--remove-side-band",
        action="store_true",
        help="Remove boxes whose centers are in the left/right side bands.",
    )
    parser.add_argument(
        "--no-clear-clean-annotations",
        action="store_true",
        help="Do not remove existing clean annotation JSON files before regeneration.",
    )
    parser.add_argument(
        "--side-band-ratio",
        type=float,
        default=None,
        help="Width ratio of left/right side bands to remove.",
    )
    args = parser.parse_args()

    logger = setup_logger("prepare_detection_data")

    cfg = load_yaml(args.config)
    det_cfg = cfg["detection_data"]
    split_cfg = det_cfg.get("split", {})
    cleaning_cfg = det_cfg.get("cleaning", {})

    seed = int(args.seed if args.seed is not None else split_cfg.get("seed", 42))
    train_ratio = float(args.train_ratio if args.train_ratio is not None else split_cfg.get("train_ratio", 0.8))
    val_ratio = float(args.val_ratio if args.val_ratio is not None else split_cfg.get("val_ratio", 0.1))
    test_ratio = args.test_ratio if args.test_ratio is not None else split_cfg.get("test_ratio", None)
    test_ratio = None if test_ratio is None else float(test_ratio)

    max_area_ratio = float(
        args.max_area_ratio
        if args.max_area_ratio is not None
        else cleaning_cfg.get("max_area_ratio", 0.25)
    )
    duplicate_iou_threshold = float(
        args.duplicate_iou_threshold
        if args.duplicate_iou_threshold is not None
        else cleaning_cfg.get("duplicate_iou_threshold", 0.90)
    )
    min_area = float(args.min_area if args.min_area is not None else cleaning_cfg.get("min_area", 4.0))
    remove_side_band = bool(args.remove_side_band or cleaning_cfg.get("remove_side_band", False))
    clear_clean_annotations = bool(
        cleaning_cfg.get("clear_clean_annotations", True)
        and not args.no_clear_clean_annotations
    )
    side_band_ratio = float(
        args.side_band_ratio
        if args.side_band_ratio is not None
        else cleaning_cfg.get("side_band_ratio", 0.10)
    )

    set_seed(seed)

    image_dir = Path(det_cfg["raw_images"])
    raw_ann_dir = Path(det_cfg["raw_annotations"])
    clean_ann_dir = Path(det_cfg["clean_annotations"])
    train_list = Path(det_cfg["train_list"])
    val_list = Path(det_cfg["val_list"])
    test_list = Path(det_cfg["test_list"])

    ensure_dir(clean_ann_dir)
    ensure_dir(train_list.parent)

    logger.info(f"Image dir: {image_dir}")
    logger.info(f"Raw annotation dir: {raw_ann_dir}")
    logger.info(f"Clean annotation dir: {clean_ann_dir}")

    if clear_clean_annotations:
        clear_clean_annotation_dir(
            clean_ann_dir=clean_ann_dir,
            data_root=Path(det_cfg["root"]),
            logger=logger,
        )

    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    if not raw_ann_dir.exists():
        raise FileNotFoundError(f"Raw annotation dir not found: {raw_ann_dir}")

    image_files = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    json_files = sorted(raw_ann_dir.glob("*.json"))

    if not json_files:
        raise RuntimeError(f"No JSON files found in: {raw_ann_dir}")

    expected_raw_images = det_cfg.get("expected_raw_images")
    expected_raw_annotations = det_cfg.get("expected_raw_annotations")

    if expected_raw_images is not None and len(image_files) < int(expected_raw_images):
        raise RuntimeError(
            f"Raw image count is lower than expected: "
            f"found={len(image_files)}, expected>={expected_raw_images}, dir={image_dir}"
        )

    if expected_raw_annotations is not None and len(json_files) < int(expected_raw_annotations):
        raise RuntimeError(
            f"Raw JSON count is lower than expected: "
            f"found={len(json_files)}, expected>={expected_raw_annotations}, dir={raw_ann_dir}"
        )

    logger.info(f"Raw image files: {len(image_files)}")
    logger.info(f"Raw JSON files: {len(json_files)}")
    logger.info(
        f"Split ratios: train={train_ratio}, val={val_ratio}, "
        f"test={test_ratio if test_ratio is not None else 1.0 - train_ratio - val_ratio}, seed={seed}"
    )
    logger.info(
        f"Cleaning params: max_area_ratio={max_area_ratio}, "
        f"duplicate_iou_threshold={duplicate_iou_threshold}, min_area={min_area}, "
        f"remove_side_band={remove_side_band}, side_band_ratio={side_band_ratio}"
    )

    valid_items = []
    missing_images = []
    failed_json = []
    empty_after_clean = []

    total_stats = {
        "num_json": len(json_files),
        "matched_images": 0,
        "missing_images": 0,
        "failed_json": 0,
        "empty_after_clean": 0,
        "raw_polygons": 0,
        "invalid_polygons": 0,
        "large_outer_boxes": 0,
        "side_band_boxes": 0,
        "duplicate_boxes": 0,
        "kept_polygons": 0,
    }

    per_file_stats = []

    for json_path in json_files:
        image_path = find_matching_image(json_path, image_dir)

        if image_path is None:
            missing_images.append(str(json_path))
            total_stats["missing_images"] += 1
            continue

        try:
            data = load_json(json_path)
            image_width, image_height = get_image_size(image_path)
            raw_polygons = extract_polygons(data)

            clean_polys, stats = clean_polygons(
                polygons=raw_polygons,
                image_width=image_width,
                image_height=image_height,
                max_area_ratio=max_area_ratio,
                duplicate_iou_threshold=duplicate_iou_threshold,
                min_area=min_area,
                remove_side_band=remove_side_band,
                side_band_ratio=side_band_ratio,
            )

        except Exception as e:
            failed_json.append(
                {
                    "json": str(json_path).replace("\\", "/"),
                    "error": repr(e),
                }
            )
            total_stats["failed_json"] += 1
            continue

        for key in [
            "raw_polygons",
            "invalid_polygons",
            "large_outer_boxes",
            "side_band_boxes",
            "duplicate_boxes",
            "kept_polygons",
        ]:
            total_stats[key] += stats[key]

        if not clean_polys:
            empty_after_clean.append(str(json_path))
            total_stats["empty_after_clean"] += 1
            continue

        clean_path = clean_ann_dir / json_path.name

        save_clean_annotation(
            output_path=clean_path,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            polygons=clean_polys,
            source_json=json_path,
        )

        valid_items.append(
            (
                str(image_path).replace("\\", "/"),
                str(clean_path).replace("\\", "/"),
            )
        )

        total_stats["matched_images"] += 1

        per_file_stats.append(
            {
                "image": str(image_path).replace("\\", "/"),
                "json": str(json_path).replace("\\", "/"),
                "clean_json": str(clean_path).replace("\\", "/"),
                "image_width": image_width,
                "image_height": image_height,
                **stats,
            }
        )

    if not valid_items:
        raise RuntimeError("No valid detection samples after cleaning.")

    train_items, val_items, test_items = split_items(
        items=valid_items,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    save_manifest(train_items, train_list)
    save_manifest(val_items, val_list)
    save_manifest(test_items, test_list)

    summary = {
        "image_dir": str(image_dir).replace("\\", "/"),
        "raw_annotation_dir": str(raw_ann_dir).replace("\\", "/"),
        "clean_annotation_dir": str(clean_ann_dir).replace("\\", "/"),
        "raw_image_files": len(image_files),
        "raw_json_files": len(json_files),
        "expected_raw_images": expected_raw_images,
        "expected_raw_annotations": expected_raw_annotations,
        "split_params": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio if test_ratio is not None else 1.0 - train_ratio - val_ratio,
        },
        "cleaning_params": {
            "clear_clean_annotations": clear_clean_annotations,
            "max_area_ratio": max_area_ratio,
            "duplicate_iou_threshold": duplicate_iou_threshold,
            "min_area": min_area,
            "remove_side_band": remove_side_band,
            "side_band_ratio": side_band_ratio,
        },
        "total_stats": total_stats,
        "valid_samples": len(valid_items),
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "test_samples": len(test_items),
        "output_files": {
            "train_list": str(train_list).replace("\\", "/"),
            "val_list": str(val_list).replace("\\", "/"),
            "test_list": str(test_list).replace("\\", "/"),
        },
        "per_file_stats": per_file_stats,
    }

    summary_path = Path(det_cfg["root"]) / "processed" / "detection_dataset_summary.json"
    save_json(summary, summary_path)

    processed_dir = Path(det_cfg["root"]) / "processed"

    remove_stale_or_write(
        missing_images,
        processed_dir / "missing_detection_images.txt",
        write_txt,
    )
    remove_stale_or_write(
        failed_json,
        processed_dir / "failed_detection_json.json",
        save_json,
    )
    remove_stale_or_write(
        empty_after_clean,
        processed_dir / "empty_after_clean.txt",
        write_txt,
    )

    logger.info(f"Raw image files: {len(image_files)}")
    logger.info(f"Raw JSON files: {len(json_files)}")
    logger.info(f"Valid samples: {len(valid_items)}")
    logger.info(f"Train: {len(train_items)}")
    logger.info(f"Val: {len(val_items)}")
    logger.info(f"Test: {len(test_items)}")
    logger.info(f"Raw polygons: {total_stats['raw_polygons']}")
    logger.info(f"Invalid polygons: {total_stats['invalid_polygons']}")
    logger.info(f"Large outer boxes removed: {total_stats['large_outer_boxes']}")
    logger.info(f"Side band boxes removed: {total_stats['side_band_boxes']}")
    logger.info(f"Duplicate boxes removed: {total_stats['duplicate_boxes']}")
    logger.info(f"Kept polygons: {total_stats['kept_polygons']}")
    logger.info(f"Saved summary to: {summary_path}")
    logger.info("Detection data preparation finished.")


if __name__ == "__main__":
    main()

