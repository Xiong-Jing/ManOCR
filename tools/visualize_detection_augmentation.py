import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from manchu_ocr.data.datasets.detection_dataset import DetectionDataset
from manchu_ocr.data.transforms.det_transforms import build_det_transform
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import ensure_dir


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().float()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    return image[:, :, ::-1].copy()


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    encoded.tofile(str(path))


def draw_polygons(image: np.ndarray, polygons: list[dict]) -> np.ndarray:
    canvas = image.copy()

    for poly in polygons:
        points = np.asarray(poly["points"], dtype=np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 255), thickness=2)

    return canvas


def get_manifest(paths_cfg: dict, split: str) -> str:
    det_cfg = paths_cfg["detection_data"]

    if split == "train":
        return det_cfg["train_list"]
    if split == "val":
        return det_cfg["val_list"]
    return det_cfg["test_list"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths_cfg = load_yaml(cfg["experiment"]["paths_config"])

    data_cfg = cfg["data"]
    aug_cfg = data_cfg.get("augmentation", {})

    transform = build_det_transform(
        target_height=int(data_cfg["target_height"]),
        target_width=int(data_cfg["target_width"]),
        keep_aspect_ratio=True,
        augment=bool(aug_cfg.get("enabled", False)),
        augmentation_cfg=aug_cfg,
    )

    dataset = DetectionDataset(
        manifest_path=get_manifest(paths_cfg, args.split),
        transform=transform,
        label_generator=None,
        check_exists=True,
    )

    exp_name = cfg["experiment"]["name"]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("outputs") / "visualizations" / "detection_augmentation" / exp_name / args.split
    )
    ensure_dir(output_dir)

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.num_samples]

    for out_idx, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        image = denormalize_image(sample["image"])
        image = draw_polygons(image, sample["polygons"])
        image_name = Path(sample["image_path"]).stem
        output_path = output_dir / f"{out_idx:03d}_{image_name}.jpg"
        imwrite_unicode(output_path, image)
        print(f"[OK] {output_path}")

    print(f"Augmentation enabled: {bool(aug_cfg.get('enabled', False))}")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
