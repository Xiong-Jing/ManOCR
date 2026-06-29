import argparse
from pathlib import Path

import cv2
import numpy as np

from manchu_ocr.pipelines.detect import DetectionPipeline
from manchu_ocr.utils.file_io import save_json
from manchu_ocr.utils.visualizer import draw_polygons, imread_unicode, imwrite_unicode


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--vis-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--binary-thresh", type=float, default=None)
    parser.add_argument("--box-thresh", type=float, default=None)
    parser.add_argument("--unclip-ratio", type=float, default=None)
    args = parser.parse_args()

    pipeline = DetectionPipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        binary_thresh=args.binary_thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
    )

    records = []
    for image_path in collect_images(Path(args.input)):
        result = pipeline(image_path)
        record = {
            "image_path": str(image_path).replace("\\", "/"),
            "boxes": result["boxes"],
            "scores": result["scores"],
        }
        records.append(record)
        print(f"{image_path}\tboxes={len(result['boxes'])}")

        if args.vis_dir:
            image = imread_unicode(image_path)
            vis = draw_polygons(image, result["boxes"], color=(0, 0, 255), thickness=2)
            out_path = Path(args.vis_dir) / f"{image_path.stem}_det.jpg"
            imwrite_unicode(out_path, vis)

    if args.output:
        save_json(records, args.output)


if __name__ == "__main__":
    main()
