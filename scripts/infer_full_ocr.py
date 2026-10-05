import argparse
from pathlib import Path

from manchu_ocr.pipelines.full_ocr_pipeline import FullOCRPipeline
from manchu_ocr.utils.file_io import ensure_dir, save_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-config", type=str, required=True)
    parser.add_argument("--det-checkpoint", type=str, required=True)
    parser.add_argument("--rec-config", type=str, required=True)
    parser.add_argument("--rec-checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/predictions/full_ocr_predictions.json")
    parser.add_argument("--text-output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.04)
    args = parser.parse_args()

    pipeline = FullOCRPipeline(
        detection_config=args.det_config,
        detection_checkpoint=args.det_checkpoint,
        recognition_config=args.rec_config,
        recognition_checkpoint=args.rec_checkpoint,
        device=args.device,
        crop_padding=args.crop_padding,
        crop_padding_ratio=args.crop_padding_ratio,
    )

    records = []
    text_output_dir = ensure_dir(args.text_output_dir) if args.text_output_dir else None

    for image_path in collect_images(Path(args.input)):
        result = pipeline(image_path)
        records.append(result)

        if text_output_dir is not None:
            text_path = text_output_dir / f"{image_path.stem}.txt"
            text_path.write_text(result.get("page_text", ""), encoding="utf-8")

        preview = result.get("flat_text", "")[:120]
        print(f"{image_path}\titems={len(result['items'])}\ttext={preview}")

    save_json(records, args.output)
    print(f"Saved predictions to: {args.output}")

    if text_output_dir is not None:
        print(f"Saved page text files to: {text_output_dir}")


if __name__ == "__main__":
    main()
