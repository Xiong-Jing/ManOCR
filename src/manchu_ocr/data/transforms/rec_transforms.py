from dataclasses import dataclass
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms


@dataclass
class RecResizePad:
    """
    Resize recognition image while preserving aspect ratio, then pad to fixed size.

    Output shape:
        [C, image_height, image_width]
    """

    image_height: int = 64
    image_width: int = 256
    keep_aspect_ratio: bool = True
    augment: bool = False
    rotation_degrees: float = 0.0
    translate_ratio: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    brightness: float = 0.0
    contrast: float = 0.0

    def __post_init__(self):
        aug_ops = []

        if self.rotation_degrees > 0 or self.translate_ratio > 0 or self.scale_min != 1.0 or self.scale_max != 1.0:
            aug_ops.append(
                transforms.RandomAffine(
                    degrees=self.rotation_degrees,
                    translate=(self.translate_ratio, self.translate_ratio),
                    scale=(self.scale_min, self.scale_max),
                    fill=(255, 255, 255),
                )
            )

        if self.brightness > 0 or self.contrast > 0:
            aug_ops.append(
                transforms.ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                )
            )

        self.augmentation = transforms.Compose(aug_ops) if aug_ops else None
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")

        if self.augment and self.augmentation is not None:
            image = self.augmentation(image)

        if self.keep_aspect_ratio:
            image = self._resize_keep_ratio(image)
        else:
            image = image.resize((self.image_width, self.image_height), Image.BILINEAR)

        tensor = self.to_tensor(image)
        tensor = self.normalize(tensor)

        return tensor

    def _resize_keep_ratio(self, image: Image.Image) -> Image.Image:
        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {image.size}")

        scale = self.image_height / float(height)
        resized_width = int(round(width * scale))
        resized_width = max(1, resized_width)
        resized_width = min(resized_width, self.image_width)

        resized = image.resize((resized_width, self.image_height), Image.BILINEAR)

        canvas = Image.new("RGB", (self.image_width, self.image_height), color=(255, 255, 255))
        canvas.paste(resized, (0, 0))

        return canvas


def build_rec_transform(
    image_height: int = 64,
    image_width: int = 256,
    keep_aspect_ratio: bool = True,
    augment: bool = False,
    augmentation_cfg: dict | None = None,
):
    augmentation_cfg = augmentation_cfg or {}

    return RecResizePad(
        image_height=image_height,
        image_width=image_width,
        keep_aspect_ratio=keep_aspect_ratio,
        augment=augment,
        rotation_degrees=float(augmentation_cfg.get("rotation_degrees", 0.0)),
        translate_ratio=float(augmentation_cfg.get("translate_ratio", 0.0)),
        scale_min=float(augmentation_cfg.get("scale_min", 1.0)),
        scale_max=float(augmentation_cfg.get("scale_max", 1.0)),
        brightness=float(augmentation_cfg.get("brightness", 0.0)),
        contrast=float(augmentation_cfg.get("contrast", 0.0)),
    )
