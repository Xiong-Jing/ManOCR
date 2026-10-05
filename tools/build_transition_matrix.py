import argparse
from pathlib import Path

import numpy as np

from manchu_ocr.utils.config import load_yaml


def read_charset(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.rstrip("\n")]


def read_labels(path: Path) -> list[str]:
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                labels.append(parts[1])
    return labels


def build_transition_matrix(labels: list[str], charset: list[str], smoothing: float = 1.0) -> np.ndarray:
    char_to_idx = {char: idx for idx, char in enumerate(charset)}
    counts = np.full((len(charset), len(charset)), smoothing, dtype=np.float64)

    for label in labels:
        for a, b in zip(label[:-1], label[1:]):
            if a in char_to_idx and b in char_to_idx:
                counts[char_to_idx[a], char_to_idx[b]] += 1.0

    counts = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
    return counts.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--charset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--smoothing", type=float, default=1.0)
    args = parser.parse_args()

    manifest = args.manifest
    charset_path = args.charset
    output = args.output

    if args.config is not None:
        cfg = load_yaml(args.config)
        rec_cfg = cfg["recognition_data"]
        manifest = manifest or rec_cfg["train_list"]
        charset_path = charset_path or rec_cfg["charset"]
        output = output or rec_cfg["transition_matrix"]

    if manifest is None or charset_path is None or output is None:
        raise ValueError("Either provide --config or provide --manifest, --charset, and --output.")

    charset = read_charset(Path(charset_path))
    labels = read_labels(Path(manifest))
    matrix = build_transition_matrix(labels, charset, smoothing=args.smoothing)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)
    print(f"[OK] saved transition matrix {matrix.shape} to {output_path}")


if __name__ == "__main__":
    main()
