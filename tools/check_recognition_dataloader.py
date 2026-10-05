import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from manchu_ocr.data.collate.rec_collate import RecCollate
from manchu_ocr.data.datasets.recognition_dataset import RecognitionDataset
from manchu_ocr.data.label_converters.ctc_label_converter import CTCLabelConverter
from manchu_ocr.data.transforms.rec_transforms import build_rec_transform
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.logger import setup_logger
from manchu_ocr.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paths/remote_server.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    logger = setup_logger("check_recognition_dataloader")

    cfg = load_yaml(args.config)
    rec_cfg = cfg["recognition_data"]

    if args.split == "train":
        manifest_path = rec_cfg["train_list"]
    elif args.split == "val":
        manifest_path = rec_cfg["val_list"]
    else:
        manifest_path = rec_cfg["test_list"]

    charset_path = rec_cfg["charset"]

    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Charset: {charset_path}")

    converter = CTCLabelConverter(charset_path)

    logger.info(f"Charset size without blank: {len(converter.chars)}")
    logger.info(f"CTC num_classes: {converter.num_classes}")
    logger.info(f"Blank index: {converter.blank_idx}")

    transform = build_rec_transform(
        image_height=args.image_height,
        image_width=args.image_width,
        keep_aspect_ratio=True,
    )

    dataset = RecognitionDataset(
        manifest_path=manifest_path,
        transform=transform,
        check_exists=True,
    )

    logger.info(f"Dataset size: {len(dataset)}")

    collate_fn = RecCollate(label_converter=converter)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(dataloader))

    images = batch["images"]
    labels = batch["labels"]
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]
    image_paths = batch["image_paths"]

    logger.info(f"images shape: {tuple(images.shape)}")
    logger.info(f"images dtype: {images.dtype}")
    logger.info(f"images min: {images.min().item():.4f}")
    logger.info(f"images max: {images.max().item():.4f}")
    logger.info(f"number of labels: {len(labels)}")
    logger.info(f"targets shape: {tuple(targets.shape)}")
    logger.info(f"target_lengths: {target_lengths.tolist()}")

    logger.info("First batch samples:")

    offset = 0
    for i, label in enumerate(labels):
        length = target_lengths[i].item()
        encoded = targets[offset: offset + length].tolist()
        offset += length

        logger.info(
            f"[{i}] path={Path(image_paths[i]).name}, "
            f"label={label}, "
            f"encoded={encoded}"
        )

    decoded_check = []
    offset = 0
    for length in target_lengths.tolist():
        encoded = targets[offset: offset + length]
        offset += length

        # 这里手动模拟一个没有 blank 的序列，用于检查 encode/decode 一致性
        decoded = "".join([converter.idx_to_char[int(idx)] for idx in encoded])
        decoded_check.append(decoded)

    logger.info(f"Decoded check: {decoded_check}")

    if decoded_check != labels:
        raise RuntimeError(
            "Encode/decode check failed. "
            f"labels={labels}, decoded={decoded_check}"
        )

    logger.info("Recognition DataLoader check passed.")


if __name__ == "__main__":
    main()

