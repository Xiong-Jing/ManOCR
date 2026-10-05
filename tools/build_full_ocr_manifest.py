import argparse
from pathlib import Path

from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paths/remote_server.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    det_cfg = cfg["detection_data"]

    manifest_key = {
        "train": "train_list",
        "val": "val_list",
        "test": "test_list",
    }[args.split]

    source = Path(det_cfg[manifest_key])

    if not source.exists():
        raise FileNotFoundError(f"Detection manifest not found: {source}")

    output = Path(args.output or f"data/manifests/full_ocr_{args.split}.txt")
    ensure_dir(output.parent)

    lines = []

    with source.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(f"Invalid detection manifest line {line_idx}: {repr(line)}")

            lines.append(line)

    if not lines:
        raise RuntimeError(f"No samples found in source manifest: {source}")

    output.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {len(lines)} full OCR samples to {output}")
    print("Note: recognition metrics require text/transcription fields in annotation JSON.")


if __name__ == "__main__":
    main()

