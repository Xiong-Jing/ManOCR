import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from manchu_ocr.data.collate.det_collate import DetCollate
from manchu_ocr.data.datasets.detection_dataset import DetectionDataset
from manchu_ocr.data.label_generators.db_label_generator import DBLabelGenerator
from manchu_ocr.data.transforms.det_transforms import build_det_transform
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import ensure_dir
from manchu_ocr.utils.logger import setup_logger


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().float()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    return image


def heatmap_to_bgr(map_: np.ndarray) -> np.ndarray:
    map_ = np.clip(map_, 0, 1)
    map_ = (map_ * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(map_, cv2.COLORMAP_JET)
    return heatmap


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    encoded.tofile(str(path))


def save_visualization(batch: dict, output_dir: Path, max_images: int = 2) -> None:
    ensure_dir(output_dir)

    images = batch["images"]
    prob_maps = batch["prob_map"]
    thresh_maps = batch["thresh_map"]
    thresh_masks = batch["thresh_mask"]
    training_masks = batch["training_mask"]
    image_paths = batch["image_paths"]

    n = min(images.shape[0], max_images)

    for i in range(n):
        image = denormalize_image(images[i])
        image_bgr = image[:, :, ::-1].copy()

        prob = prob_maps[i, 0].detach().cpu().numpy()
        thresh = thresh_maps[i, 0].detach().cpu().numpy()
        thresh_mask = thresh_masks[i, 0].detach().cpu().numpy()
        train_mask = training_masks[i, 0].detach().cpu().numpy()

        prob_color = heatmap_to_bgr(prob)
        thresh_color = heatmap_to_bgr((thresh - 0.3) / 0.4)
        thresh_mask_color = heatmap_to_bgr(thresh_mask)
        train_mask_color = heatmap_to_bgr(train_mask)

        overlay = cv2.addWeighted(image_bgr, 0.65, prob_color, 0.35, 0)

        stem = Path(image_paths[i]).stem

        imwrite_unicode(output_dir / f"{i:03d}_{stem}_image.jpg", image_bgr)
        imwrite_unicode(output_dir / f"{i:03d}_{stem}_prob.jpg", prob_color)
        imwrite_unicode(output_dir / f"{i:03d}_{stem}_prob_overlay.jpg", overlay)
        imwrite_unicode(output_dir / f"{i:03d}_{stem}_thresh.jpg", thresh_color)
        imwrite_unicode(output_dir / f"{i:03d}_{stem}_thresh_mask.jpg", thresh_mask_color)
        imwrite_unicode(output_dir / f"{i:03d}_{stem}_training_mask.jpg", train_mask_color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paths/remote_server.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=1056)
    parser.add_argument("--target-width", type=int, default=768)
    parser.add_argument("--shrink-ratio", type=float, default=0.4)
    parser.add_argument("--use-asymmetric-shrink", action="store_true")
    parser.add_argument("--shrink-ratio-x", type=float, default=0.65)
    parser.add_argument("--shrink-ratio-y", type=float, default=0.90)
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args()

    logger = setup_logger("check_db_label_generator")

    cfg = load_yaml(args.config)
    det_cfg = cfg["detection_data"]

    if args.split == "train":
        manifest_path = det_cfg["train_list"]
    elif args.split == "val":
        manifest_path = det_cfg["val_list"]
    else:
        manifest_path = det_cfg["test_list"]

    transform = build_det_transform(
        target_height=args.target_height,
        target_width=args.target_width,
        keep_aspect_ratio=True,
    )

    label_generator = DBLabelGenerator(
        shrink_ratio=args.shrink_ratio,
        thresh_min=0.3,
        thresh_max=0.7,
        min_text_size=3,
        use_asymmetric_shrink=args.use_asymmetric_shrink,
        shrink_ratio_x=args.shrink_ratio_x,
        shrink_ratio_y=args.shrink_ratio_y,
    )

    dataset = DetectionDataset(
        manifest_path=manifest_path,
        transform=transform,
        label_generator=label_generator,
        check_exists=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DetCollate(),
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(dataloader))

    logger.info(f"images shape: {tuple(batch['images'].shape)}")
    logger.info(f"prob_map shape: {tuple(batch['prob_map'].shape)}")
    logger.info(f"thresh_map shape: {tuple(batch['thresh_map'].shape)}")
    logger.info(f"thresh_mask shape: {tuple(batch['thresh_mask'].shape)}")
    logger.info(f"training_mask shape: {tuple(batch['training_mask'].shape)}")
    logger.info(f"num_valid_polygons: {batch['num_valid_polygons'].tolist()}")

    logger.info(
        f"prob_map min/max: "
        f"{batch['prob_map'].min().item():.4f}/"
        f"{batch['prob_map'].max().item():.4f}"
    )

    logger.info(
        f"thresh_map min/max: "
        f"{batch['thresh_map'].min().item():.4f}/"
        f"{batch['thresh_map'].max().item():.4f}"
    )

    if args.save_vis:
        output_dir = (
            Path(det_cfg["root"])
            / "processed"
            / "visualized_db_labels"
            / args.split
        )

        save_visualization(batch, output_dir=output_dir, max_images=args.batch_size)
        logger.info(f"Saved visualization to: {output_dir}")

    logger.info("DB label generator check passed.")


if __name__ == "__main__":
    main()

