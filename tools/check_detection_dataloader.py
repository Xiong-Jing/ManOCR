import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from manchu_ocr.data.collate.det_collate import DetCollate
from manchu_ocr.data.datasets.detection_dataset import DetectionDataset
from manchu_ocr.data.transforms.det_transforms import build_det_transform
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import ensure_dir
from manchu_ocr.utils.logger import setup_logger
from manchu_ocr.utils.seed import set_seed


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor [C,H,W] to uint8 BGR image for OpenCV saving.
    """
    image = tensor.detach().cpu().float()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    # RGB to BGR
    image = image[:, :, ::-1].copy()

    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix
    if ext == "":
        ext = ".jpg"

    ok, encoded = cv2.imencode(ext, image)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    encoded.tofile(str(path))


def draw_batch(batch: dict, output_dir: Path, max_images: int = 4) -> None:
    ensure_dir(output_dir)

    images = batch["images"]
    polygons_batch = batch["polygons"]
    image_paths = batch["image_paths"]

    n = min(images.shape[0], max_images)

    for i in range(n):
        image = denormalize_image(images[i])
        polygons = polygons_batch[i]

        for poly in polygons:
            points = np.array(poly["points"], dtype=np.int32)
            cv2.polylines(
                image,
                [points],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2,
            )

        image_name = Path(image_paths[i]).stem
        out_path = output_dir / f"{i:03d}_{image_name}.jpg"
        imwrite_unicode(out_path, image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paths/remote_server.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=1056)
    parser.add_argument("--target-width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    logger = setup_logger("check_detection_dataloader")

    cfg = load_yaml(args.config)
    det_cfg = cfg["detection_data"]

    if args.split == "train":
        manifest_path = det_cfg["train_list"]
    elif args.split == "val":
        manifest_path = det_cfg["val_list"]
    else:
        manifest_path = det_cfg["test_list"]

    logger.info(f"Manifest: {manifest_path}")

    transform = build_det_transform(
        target_height=args.target_height,
        target_width=args.target_width,
        keep_aspect_ratio=True,
    )

    dataset = DetectionDataset(
        manifest_path=manifest_path,
        transform=transform,
        check_exists=True,
    )

    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DetCollate(),
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(dataloader))

    images = batch["images"]
    polygons = batch["polygons"]
    metas = batch["metas"]

    logger.info(f"images shape: {tuple(images.shape)}")
    logger.info(f"images dtype: {images.dtype}")
    logger.info(f"images min: {images.min().item():.4f}")
    logger.info(f"images max: {images.max().item():.4f}")

    for i in range(len(polygons)):
        logger.info(
            f"[{i}] file={Path(batch['image_paths'][i]).name}, "
            f"num_polygons={len(polygons[i])}, "
            f"meta={json.dumps(metas[i], ensure_ascii=False)}"
        )

        if len(polygons[i]) > 0:
            logger.info(f"first polygon: {polygons[i][0]}")

    if args.save_vis:
        output_dir = (
            Path(det_cfg["root"])
            / "processed"
            / "visualized_dataloader"
            / args.split
        )

        draw_batch(batch, output_dir=output_dir, max_images=args.batch_size)
        logger.info(f"Saved visualization to: {output_dir}")

    logger.info("Detection DataLoader check passed.")


if __name__ == "__main__":
    main()

