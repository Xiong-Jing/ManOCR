import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import ensure_dir


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def imread_unicode(path: Path):
    """
    Unicode-safe image reading for Windows.
    cv2.imread may fail when the path contains Chinese characters.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Failed to decode image: {path}")

    return image


def imwrite_unicode(path: Path, image) -> None:
    """
    Unicode-safe image writing for Windows.
    cv2.imwrite may fail when the output path contains Chinese characters.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix
    if ext == "":
        ext = ".jpg"

    success, encoded = cv2.imencode(ext, image)

    if not success:
        raise RuntimeError(f"Failed to encode image for saving: {path}")

    encoded.tofile(str(path))


def draw_annotation(image_path: Path, ann_path: Path, output_path: Path) -> None:
    image = imread_unicode(image_path)

    ann = load_json(ann_path)

    for poly in ann["polygons"]:
        points = poly["points"]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    imwrite_unicode(output_path, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paths/remote_server.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    det_cfg = cfg["detection_data"]

    if args.split == "train":
        manifest_path = Path(det_cfg["train_list"])
    elif args.split == "val":
        manifest_path = Path(det_cfg["val_list"])
    else:
        manifest_path = Path(det_cfg["test_list"])

    output_dir = Path(det_cfg["root"]) / "processed" / "visualized_clean" / args.split
    ensure_dir(output_dir)

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    items = []

    for line_idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        parts = line.split("\t")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid manifest line {line_idx}: {repr(line)}"
            )

        image_path, ann_path = parts
        items.append((Path(image_path), Path(ann_path)))

    if len(items) == 0:
        raise RuntimeError(f"No items found in manifest: {manifest_path}")

    random.Random(args.seed).shuffle(items)
    items = items[: args.num_samples]

    for idx, (image_path, ann_path) in enumerate(items):
        safe_stem = f"{idx:03d}_{image_path.stem}"
        output_path = output_dir / f"{safe_stem}.jpg"

        draw_annotation(image_path, ann_path, output_path)
        print(f"[OK] {output_path}")


if __name__ == "__main__":
    main()

