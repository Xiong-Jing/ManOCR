import argparse
from pathlib import Path

from manchu_ocr.pipelines.recognize import RecognitionPipeline
from manchu_ocr.utils.file_io import save_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or image directory.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pipeline = RecognitionPipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    records = []
    for image_path in collect_images(Path(args.input)):
        result = pipeline(image_path)
        records.append(
            {
                "image_path": str(image_path).replace("\\", "/"),
                **result,
            }
        )
        print(f"{image_path}\t{result['text']}\t{result['confidence']:.4f}")

    if args.output:
        save_json(records, args.output)


if __name__ == "__main__":
    main()
