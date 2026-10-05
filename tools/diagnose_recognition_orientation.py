import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader

from manchu_ocr.data.collate.rec_collate import RecCollate
from manchu_ocr.data.datasets.recognition_dataset import RecognitionDataset
from manchu_ocr.data.label_converters.ctc_label_converter import CTCLabelConverter
from manchu_ocr.data.transforms.rec_transforms import build_rec_transform
from manchu_ocr.metrics.recognition_metrics import compute_recognition_metrics
from manchu_ocr.models.recognition.builder import build_recognition_model
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import save_json
from manchu_ocr.utils.logger import setup_logger


ORIENTATION_MODES = ["original", "flip_top_bottom", "flip_left_right", "rotate_180"]


class OrientationTransform:
    def __init__(
        self,
        image_height: int,
        image_width: int,
        mode: str = "original",
    ):
        if mode not in ORIENTATION_MODES:
            raise ValueError(f"Unsupported orientation mode: {mode}")

        self.mode = mode
        self.base_transform = build_rec_transform(
            image_height=image_height,
            image_width=image_width,
            keep_aspect_ratio=True,
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")

        if self.mode == "flip_top_bottom":
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif self.mode == "flip_left_right":
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif self.mode == "rotate_180":
            image = image.transpose(Image.Transpose.ROTATE_180)

        return self.base_transform(image)


def build_device(cfg: Dict) -> torch.device:
    runtime_cfg = cfg.get("runtime", {})
    requested_device = runtime_cfg.get("device", "auto")
    allow_cpu = bool(runtime_cfg.get("allow_cpu", True))

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requires CUDA, but CUDA is unavailable.")
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError("Diagnosis is running on CPU, but allow_cpu=false.")

    return device


def build_dataloader(
    manifest_path: str,
    charset_path: str,
    image_height: int,
    image_width: int,
    mode: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, CTCLabelConverter]:
    converter = CTCLabelConverter(charset_path)
    transform = OrientationTransform(
        image_height=image_height,
        image_width=image_width,
        mode=mode,
    )

    dataset = RecognitionDataset(
        manifest_path=manifest_path,
        transform=transform,
        check_exists=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=RecCollate(label_converter=converter),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataloader, converter


@torch.no_grad()
def evaluate_orientation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    converter: CTCLabelConverter,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    all_preds = []
    all_labels = []

    empty_predictions = 0
    total_pred_len = 0
    total_label_len = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device, non_blocking=True)
        logits = model(images).float()
        preds = converter.decode_logits(logits.detach().cpu())
        labels = batch["labels"]

        all_preds.extend(preds)
        all_labels.extend(labels)

        empty_predictions += sum(1 for pred in preds if pred == "")
        total_pred_len += sum(len(pred) for pred in preds)
        total_label_len += sum(len(label) for label in labels)
        total_samples += len(labels)

    metrics = compute_recognition_metrics(all_preds, all_labels)
    metrics["num_samples"] = total_samples
    metrics["empty_prediction_ratio"] = empty_predictions / max(total_samples, 1)
    metrics["avg_pred_len"] = total_pred_len / max(total_samples, 1)
    metrics["avg_label_len"] = total_label_len / max(total_samples, 1)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths_cfg = load_yaml(cfg["experiment"]["paths_config"])
    logger = setup_logger("diagnose_recognition_orientation")

    device = build_device(cfg)

    rec_cfg = paths_cfg["recognition_data"]
    if args.split == "train":
        manifest_path = rec_cfg["train_list"]
    elif args.split == "val":
        manifest_path = rec_cfg["val_list"]
    else:
        manifest_path = rec_cfg["test_list"]

    image_height = int(cfg["data"]["image_height"])
    image_width = int(cfg["data"]["image_width"])
    batch_size = args.batch_size or int(cfg["data"]["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["data"].get("num_workers", 0))

    model = build_recognition_model(cfg, num_classes=CTCLabelConverter(rec_cfg["charset"]).num_classes).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Manifest: {manifest_path}")

    results = {}

    for mode in ORIENTATION_MODES:
        dataloader, converter = build_dataloader(
            manifest_path=manifest_path,
            charset_path=rec_cfg["charset"],
            image_height=image_height,
            image_width=image_width,
            mode=mode,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        metrics = evaluate_orientation(
            model=model,
            dataloader=dataloader,
            converter=converter,
            device=device,
            max_batches=args.max_batches,
        )

        results[mode] = metrics
        logger.info(
            f"{mode}: "
            f"WA={metrics['word_accuracy']:.4f}, "
            f"CA={metrics['character_accuracy']:.4f}, "
            f"CER={metrics['cer']:.4f}, "
            f"empty={metrics['empty_prediction_ratio']:.4f}, "
            f"pred_len={metrics['avg_pred_len']:.2f}, "
            f"label_len={metrics['avg_label_len']:.2f}"
        )

    output = {
        "config": args.config,
        "checkpoint": str(checkpoint_path).replace("\\", "/"),
        "split": args.split,
        "max_batches": args.max_batches,
        "results": results,
    }

    if args.output is not None:
        save_json(output, args.output)
        logger.info(f"Saved diagnosis to: {args.output}")

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
