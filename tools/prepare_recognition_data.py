import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import ensure_dir, save_json, write_txt
from manchu_ocr.utils.logger import setup_logger
from manchu_ocr.utils.seed import set_seed


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VALID_LABEL_PATTERN = re.compile(r"^[a-z’'\-]+$")
IMAGE_ID_PAD_WIDTHS = [6]


def normalize_path_str(path: str) -> str:
    return str(Path(path)).replace("\\", "/")


def normalize_label(label: str) -> str:
    """
    Normalize Romanized Manchu transcription labels.

    Rules:
    - convert to string
    - strip leading/trailing spaces
    - remove internal whitespace
    - convert to lower-case
    - normalize curly apostrophe variants
    """
    label = str(label)
    label = label.strip()
    label = re.sub(r"\s+", "", label)
    label = label.lower()

    # Normalize apostrophe variants.
    label = label.replace("‘", "’").replace("`", "’").replace("ʼ", "’")

    return label


def find_column(columns: List[str], candidates: List[str]) -> str | None:
    lower_map = {str(c).lower().strip(): c for c in columns}

    for cand in candidates:
        cand_lower = cand.lower().strip()
        if cand_lower in lower_map:
            return lower_map[cand_lower]

    for col in columns:
        col_lower = str(col).lower().strip()
        for cand in candidates:
            if cand.lower().strip() in col_lower:
                return col

    return None


def infer_excel_columns(df: pd.DataFrame) -> Tuple[str, str]:
    columns = list(df.columns)

    image_col = find_column(
        columns,
        [
            "image",
            "img",
            "image_path",
            "img_path",
            "filename",
            "file",
            "name",
        ],
    )

    label_col = find_column(
        columns,
        [
            "label",
            "text",
            "transcription",
            "roman",
            "romanization",
            "romaji",
        ],
    )

    if image_col is None or label_col is None:
        if len(columns) < 2:
            raise ValueError(
                f"Excel needs at least two columns: image path/name and label. Current columns: {columns}"
            )
        image_col = columns[0]
        label_col = columns[1]

    return image_col, label_col


def build_image_index(image_root: Path) -> Dict[str, Path]:
    """
    Build index for recognition word images.

    Index keys:
    - full filename, e.g. word_000001.jpg
    - stem, e.g. word_000001
    """
    image_index: Dict[str, Path] = {}

    for path in image_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        image_index[path.name] = path
        image_index[path.stem] = path

    return image_index


def make_image_id_candidates(raw_value: str) -> List[str]:
    """
    Generate possible image id candidates.

    Example:
        Excel value: 76080
        Candidates:
            76080
            076080
            76080.jpg
            076080.jpg
    """
    value = str(raw_value).strip()

    if value == "" or value.lower() == "nan":
        return []

    # Handle numeric IDs read by pandas as values such as 76080.0.
    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]

    # Strip directories and keep only the filename.
    value = Path(value).name.strip()

    candidates = []

    def add(x: str) -> None:
        if x and x not in candidates:
            candidates.append(x)

    add(value)

    stem = Path(value).stem
    suffix = Path(value).suffix

    if stem:
        add(stem)

    # If the stem is numeric, add zero-padded variants.
    if stem.isdigit():
        add(str(int(stem)))  # non-padded variant

        for width in IMAGE_ID_PAD_WIDTHS:
            add(stem.zfill(width))
            add(str(int(stem)).zfill(width))

    # Add common image suffixes to all stem candidates.
    base_candidates = candidates[:]
    for item in base_candidates:
        item_stem = Path(item).stem
        item_suffix = Path(item).suffix

        if item_suffix == "":
            for ext in IMAGE_EXTS:
                add(item_stem + ext)

    # Keep the original value when it already has a suffix.
    if suffix:
        add(value)

    return candidates


def resolve_image_path(raw_value: str, image_root: Path, image_index: Dict[str, Path]) -> Path | None:
    candidates = make_image_id_candidates(raw_value)

    for candidate in candidates:
        candidate_path = Path(candidate)

        # 1. Absolute path.
        if candidate_path.is_absolute() and candidate_path.exists():
            return candidate_path

        # 2. Direct path relative to raw_images.
        direct = image_root / candidate
        if direct.exists():
            return direct

        # 3. Lookup in the filename index.
        if candidate in image_index:
            return image_index[candidate]

        # 4. Lookup again by stem.
        stem = Path(candidate).stem
        if stem in image_index:
            return image_index[stem]

    return None


def check_image_readable(path: Path) -> Tuple[bool, Tuple[int, int] | None]:
    try:
        with Image.open(path) as img:
            return True, img.size
    except Exception:
        return False, None


def split_samples(
    samples: List[Tuple[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    samples = samples[:]
    random.Random(seed).shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    return train_samples, val_samples, test_samples


def build_charset(labels: List[str]) -> List[str]:
    chars = sorted(set("".join(labels)))
    return chars


def build_transition_matrix(labels: List[str], charset: List[str], smoothing: float = 1.0) -> np.ndarray:
    """
    Build character transition probability matrix.

    T[i, j] means probability of char_j appearing after char_i.
    """
    char_to_idx = {c: i for i, c in enumerate(charset)}
    k = len(charset)

    counts = np.full((k, k), smoothing, dtype=np.float64)

    for label in labels:
        if len(label) < 2:
            continue

        for a, b in zip(label[:-1], label[1:]):
            if a in char_to_idx and b in char_to_idx:
                counts[char_to_idx[a], char_to_idx[b]] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    matrix = counts / np.maximum(row_sums, 1e-12)

    return matrix.astype(np.float32)


def save_manifest(samples: List[Tuple[str, str]], path: Path) -> None:
    lines = [f"{normalize_path_str(img)}\t{label}" for img, label in samples]
    write_txt(lines, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paths/remote_server.yaml",
        help="Path to paths yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Try opening every image. Slower but safer.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    logger = setup_logger("prepare_recognition_data")
    cfg = load_yaml(args.config)

    rec_cfg = cfg["recognition_data"]

    image_root = Path(rec_cfg["raw_images"])
    excel_path = Path(rec_cfg["raw_excel"])

    labels_csv = Path(rec_cfg["labels_csv"])
    charset_path = Path(rec_cfg["charset"])
    transition_matrix_path = Path(rec_cfg["transition_matrix"])
    train_list = Path(rec_cfg["train_list"])
    val_list = Path(rec_cfg["val_list"])
    test_list = Path(rec_cfg["test_list"])

    processed_root = labels_csv.parent
    ensure_dir(processed_root)

    logger.info(f"Recognition image root: {image_root}")
    logger.info(f"Recognition Excel: {excel_path}")

    if not image_root.exists():
        raise FileNotFoundError(f"Recognition image root not found: {image_root}")

    if not excel_path.exists():
        raise FileNotFoundError(f"Recognition Excel not found: {excel_path}")

    logger.info("Reading Excel...")
    df = pd.read_excel(excel_path)

    image_col, label_col = infer_excel_columns(df)
    logger.info(f"Using image column: {image_col}")
    logger.info(f"Using label column: {label_col}")

    logger.info("Indexing images...")
    image_index = build_image_index(image_root)
    logger.info(f"Indexed image keys: {len(image_index)}")

    samples: List[Tuple[str, str]] = []
    missing_images = []
    empty_labels = []
    invalid_labels = []
    unreadable_images = []
    image_sizes = []

    for idx, row in df.iterrows():
        raw_image = row[image_col]
        raw_label = row[label_col]

        label = normalize_label(raw_label)

        if label == "" or label.lower() == "nan":
            empty_labels.append(int(idx))
            continue

        if not VALID_LABEL_PATTERN.fullmatch(label):
            invalid_labels.append(
                {
                    "row": int(idx) + 2,
                    "image": str(raw_image),
                    "label": label,
                }
            )
            continue

        image_path = resolve_image_path(str(raw_image), image_root, image_index)

        if image_path is None:
            missing_images.append(str(raw_image))
            continue

        if args.check_images:
            ok, size = check_image_readable(image_path)
            if not ok:
                unreadable_images.append(str(image_path))
                continue
            if size is not None:
                image_sizes.append(size)

        samples.append((str(image_path), label))

    if len(samples) == 0:
        raise RuntimeError("No valid recognition samples found. Please check Excel columns and image paths.")

    logger.info(f"Valid samples: {len(samples)}")
    logger.info(f"Missing images: {len(missing_images)}")
    logger.info(f"Empty labels: {len(empty_labels)}")
    logger.info(f"Invalid labels:{len(invalid_labels)}")
    logger.info(f"Unreadable images: {len(unreadable_images)}")

    labels = [label for _, label in samples]
    charset = build_charset(labels)

    logger.info(f"Charset size without CTC blank: {len(charset)}")
    logger.info(f"Charset: {''.join(charset)}")

    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    logger.info(f"Test samples: {len(test_samples)}")

    logger.info("Saving labels CSV...")
    output_df = pd.DataFrame(samples, columns=["image_path", "label"])
    output_df.to_csv(labels_csv, index=False, encoding="utf-8-sig")

    normalized_csv = processed_root / "labels_normalized.csv"
    output_df.to_csv(normalized_csv, index=False, encoding="utf-8-sig")

    logger.info("Saving manifests...")
    save_manifest(train_samples, train_list)
    save_manifest(val_samples, val_list)
    save_manifest(test_samples, test_list)

    logger.info("Saving charset...")
    write_txt(charset, charset_path)

    logger.info("Building transition matrix from training labels...")
    train_labels = [label for _, label in train_samples]
    transition_matrix = build_transition_matrix(
        labels=train_labels,
        charset=charset,
        smoothing=args.smoothing,
    )

    transition_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(transition_matrix_path, transition_matrix)

    label_lengths = [len(label) for label in labels]
    char_counter = Counter("".join(labels))

    summary = {
        "excel_path": normalize_path_str(excel_path),
        "image_root": normalize_path_str(image_root),
        "image_column": str(image_col),
        "label_column": str(label_col),
        "valid_samples": len(samples),
        "missing_images": len(missing_images),
        "empty_labels": len(empty_labels),
        "invalid_labels": len(invalid_labels),
        "unreadable_images": len(unreadable_images),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "charset_size_without_ctc_blank": len(charset),
        "charset": charset,
        "label_length": {
            "min": int(np.min(label_lengths)),
            "max": int(np.max(label_lengths)),
            "mean": float(np.mean(label_lengths)),
            "median": float(np.median(label_lengths)),
        },
        "top_50_chars": char_counter.most_common(50),
        "transition_matrix_shape": list(transition_matrix.shape),
        "output_files": {
            "labels_csv": normalize_path_str(labels_csv),
            "labels_normalized_csv": normalize_path_str(normalized_csv),
            "charset": normalize_path_str(charset_path),
            "transition_matrix": normalize_path_str(transition_matrix_path),
            "train_list": normalize_path_str(train_list),
            "val_list": normalize_path_str(val_list),
            "test_list": normalize_path_str(test_list),
        },
    }

    if image_sizes:
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]
        summary["image_size"] = {
            "width_min": int(np.min(widths)),
            "width_max": int(np.max(widths)),
            "width_mean": float(np.mean(widths)),
            "height_min": int(np.min(heights)),
            "height_max": int(np.max(heights)),
            "height_mean": float(np.mean(heights)),
        }

    summary_path = processed_root / "dataset_summary.json"
    save_json(summary, summary_path)

    if missing_images:
        write_txt(missing_images[:1000], processed_root / "missing_images_first_1000.txt")
    else:
        missing_file = processed_root / "missing_images_first_1000.txt"
        if missing_file.exists():
            missing_file.unlink()

    if invalid_labels:
        save_json(invalid_labels[:1000], processed_root / "invalid_labels_first_1000.json")
    else:
        invalid_file = processed_root / "invalid_labels_first_1000.json"
        if invalid_file.exists():
            invalid_file.unlink()

    if unreadable_images:
        write_txt(unreadable_images, processed_root / "unreadable_images.txt")
    else:
        unreadable_file = processed_root / "unreadable_images.txt"
        if unreadable_file.exists():
            unreadable_file.unlink()

    logger.info(f"Saved summary: {summary_path}")
    logger.info("Recognition data preparation finished.")


if __name__ == "__main__":
    main()


