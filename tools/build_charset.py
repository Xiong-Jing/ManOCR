import argparse
import csv
from collections import Counter
from pathlib import Path

from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import save_json, write_txt


def read_labels_from_manifest(path: Path) -> list[str]:
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                labels.append(parts[1])
    return labels


def read_labels_from_csv(path: Path) -> list[str]:
    labels = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return labels

        label_key = None
        for candidate in ["label", "text", "transcription"]:
            if candidate in reader.fieldnames:
                label_key = candidate
                break

        if label_key is None:
            raise ValueError(f"Could not find label column in CSV: {path}")

        for row in reader:
            label = str(row.get(label_key, "")).strip()
            if label:
                labels.append(label)

    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--summary", type=str, default=None)
    args = parser.parse_args()

    manifest = args.manifest
    output = args.output
    summary = args.summary

    if args.config is not None:
        cfg = load_yaml(args.config)
        rec_cfg = cfg["recognition_data"]
        manifest = manifest or rec_cfg.get("labels_csv") or rec_cfg["train_list"]
        output = output or rec_cfg["charset"]
        summary = summary or str(Path(rec_cfg["charset"]).with_name("charset_summary.json"))

    if manifest is None or output is None:
        raise ValueError("Either provide --config or provide both --manifest and --output.")

    manifest_path = Path(manifest)
    if manifest_path.suffix.lower() == ".csv":
        labels = read_labels_from_csv(manifest_path)
    else:
        labels = read_labels_from_manifest(manifest_path)
    chars = sorted(set("".join(labels)))
    write_txt(chars, output)

    if summary:
        counter = Counter("".join(labels))
        save_json(
            {
                "num_labels": len(labels),
                "charset_size": len(chars),
                "charset": chars,
                "char_counts": counter.most_common(),
            },
            summary,
        )

    print(f"[OK] charset_size={len(chars)} saved to {output}")


if __name__ == "__main__":
    main()
