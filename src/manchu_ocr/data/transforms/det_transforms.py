from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _clip_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    points[:, 0] = np.clip(points[:, 0], 0, max(width - 1, 0))
    points[:, 1] = np.clip(points[:, 1], 0, max(height - 1, 0))
    return points


def _update_polygon_points(poly: Dict, points: np.ndarray, width: int, height: int) -> Dict:
    points = _clip_points(points.astype(np.float32), width=width, height=height)
    xs = points[:, 0]
    ys = points[:, 1]

    updated = {
        **poly,
        "points": [[float(x), float(y)] for x, y in points],
        "bbox": [
            float(xs.min()),
            float(ys.min()),
            float(xs.max()),
            float(ys.max()),
        ],
    }
    if "box" in poly:
        updated["box"] = updated["points"]
    return updated


def _odd_kernel_size(size: int) -> int:
    size = int(size)
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    return size


def _sample_kernel_size(values: List[int] | Tuple[int, ...], default: int) -> int:
    if not values:
        return _odd_kernel_size(default)
    return _odd_kernel_size(random.choice(list(values)))


def _motion_blur_kernel(kernel_size: int) -> np.ndarray:
    kernel_size = _odd_kernel_size(kernel_size)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    direction = random.choice(["horizontal", "vertical", "diag", "anti_diag"])

    if direction == "horizontal":
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "vertical":
        kernel[:, kernel_size // 2] = 1.0
    elif direction == "diag":
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1.0)

    return kernel / kernel.sum()


def _defocus_kernel(kernel_size: int) -> np.ndarray:
    kernel_size = _odd_kernel_size(kernel_size)
    radius = kernel_size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (xx * xx + yy * yy) <= radius * radius
    kernel = mask.astype(np.float32)
    return kernel / max(float(kernel.sum()), 1.0)


@dataclass
class DetResizePad:
    """
    Resize detection image while preserving aspect ratio, then pad to fixed size.

    When augment=True, training-only degradations are applied before resizing:
        - non-uniform vertical stretch / horizontal compression
        - elastic distortion
        - blur degradation
        - multiplicative and Gaussian noise
        - forced contrast reduction
    """

    target_height: int = 1056
    target_width: int = 768
    keep_aspect_ratio: bool = True
    augment: bool = False
    augmentation_cfg: Dict | None = None

    def __post_init__(self):
        self.augmentation_cfg = self.augmentation_cfg or {}
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _apply_nonuniform_scale(
        self,
        image: Image.Image,
        polygons: List[Dict],
    ) -> Tuple[Image.Image, List[Dict], Dict]:
        cfg = self.augmentation_cfg
        prob = float(cfg.get("nonuniform_scale_prob", 0.0))

        if random.random() >= prob:
            return image, polygons, {"nonuniform_scale": False}

        width, height = image.size
        scale_x = random.uniform(
            float(cfg.get("horizontal_scale_min", 0.86)),
            float(cfg.get("horizontal_scale_max", 0.96)),
        )
        scale_y = random.uniform(
            float(cfg.get("vertical_scale_min", 1.05)),
            float(cfg.get("vertical_scale_max", 1.18)),
        )

        offset_x = (width - width * scale_x) / 2.0
        offset_y = (height - height * scale_y) / 2.0

        matrix = np.asarray(
            [
                [scale_x, 0.0, offset_x],
                [0.0, scale_y, offset_y],
            ],
            dtype=np.float32,
        )

        image_np = np.asarray(image, dtype=np.uint8)
        warped = cv2.warpAffine(
            image_np,
            matrix,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        new_polygons = []
        for poly in polygons:
            points = np.asarray(poly["points"], dtype=np.float32)
            ones = np.ones((points.shape[0], 1), dtype=np.float32)
            homo = np.concatenate([points, ones], axis=1)
            new_points = homo @ matrix.T
            new_polygons.append(_update_polygon_points(poly, new_points, width, height))

        return (
            Image.fromarray(warped),
            new_polygons,
            {
                "nonuniform_scale": True,
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
            },
        )

    def _apply_elastic_distortion(
        self,
        image: Image.Image,
        polygons: List[Dict],
    ) -> Tuple[Image.Image, List[Dict], Dict]:
        cfg = self.augmentation_cfg
        prob = float(cfg.get("elastic_prob", 0.0))

        if random.random() >= prob:
            return image, polygons, {"elastic": False}

        width, height = image.size
        alpha_x = float(cfg.get("elastic_alpha_x", cfg.get("elastic_alpha", 12.0)))
        alpha_y = float(cfg.get("elastic_alpha_y", cfg.get("elastic_alpha", 8.0)))
        sigma = float(cfg.get("elastic_sigma", 8.0))

        dx = (np.random.rand(height, width).astype(np.float32) * 2.0 - 1.0)
        dy = (np.random.rand(height, width).astype(np.float32) * 2.0 - 1.0)
        dx = cv2.GaussianBlur(dx, ksize=(0, 0), sigmaX=sigma) * alpha_x
        dy = cv2.GaussianBlur(dy, ksize=(0, 0), sigmaX=sigma) * alpha_y

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )

        # map = destination -> source. A source point approximately moves by +dx/+dy.
        map_x = grid_x - dx
        map_y = grid_y - dy

        image_np = np.asarray(image, dtype=np.uint8)
        warped = cv2.remap(
            image_np,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        new_polygons = []
        for poly in polygons:
            points = np.asarray(poly["points"], dtype=np.float32)
            sample_x = np.clip(np.round(points[:, 0]).astype(np.int32), 0, width - 1)
            sample_y = np.clip(np.round(points[:, 1]).astype(np.int32), 0, height - 1)
            new_points = points.copy()
            new_points[:, 0] = new_points[:, 0] + dx[sample_y, sample_x]
            new_points[:, 1] = new_points[:, 1] + dy[sample_y, sample_x]
            new_polygons.append(_update_polygon_points(poly, new_points, width, height))

        return (
            Image.fromarray(warped),
            new_polygons,
            {
                "elastic": True,
                "alpha_x": float(alpha_x),
                "alpha_y": float(alpha_y),
                "sigma": float(sigma),
            },
        )

    def _apply_blur(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        cfg = self.augmentation_cfg
        prob = float(cfg.get("blur_prob", 0.0))

        if random.random() >= prob:
            return image, {"blur": "none"}

        blur_types = ["gaussian", "motion", "defocus"]
        weights = [
            float(cfg.get("gaussian_blur_weight", 0.40)),
            float(cfg.get("motion_blur_weight", 0.35)),
            float(cfg.get("defocus_blur_weight", 0.25)),
        ]
        blur_type = random.choices(blur_types, weights=weights, k=1)[0]

        image_np = np.asarray(image, dtype=np.uint8)

        if blur_type == "gaussian":
            k = _sample_kernel_size(cfg.get("gaussian_kernel_sizes", [3, 5, 7]), 5)
            sigma = random.uniform(
                float(cfg.get("gaussian_sigma_min", 0.4)),
                float(cfg.get("gaussian_sigma_max", 1.4)),
            )
            blurred = cv2.GaussianBlur(image_np, (k, k), sigmaX=sigma)
            meta = {"blur": "gaussian", "kernel": int(k), "sigma": float(sigma)}
        elif blur_type == "motion":
            k = _sample_kernel_size(cfg.get("motion_kernel_sizes", [5, 9, 13]), 9)
            kernel = _motion_blur_kernel(k)
            blurred = cv2.filter2D(image_np, -1, kernel)
            meta = {"blur": "motion", "kernel": int(k)}
        else:
            k = _sample_kernel_size(cfg.get("defocus_kernel_sizes", [5, 9, 13]), 9)
            kernel = _defocus_kernel(k)
            blurred = cv2.filter2D(image_np, -1, kernel)
            meta = {"blur": "defocus", "kernel": int(k)}

        return Image.fromarray(blurred), meta

    def _apply_noise_and_contrast(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        cfg = self.augmentation_cfg
        prob = float(cfg.get("noise_contrast_prob", 0.0))

        if random.random() >= prob:
            return image, {"noise_contrast": False}

        image_np = np.asarray(image, dtype=np.float32)

        contrast = random.uniform(
            float(cfg.get("contrast_min", 0.50)),
            float(cfg.get("contrast_max", 0.78)),
        )
        mean = image_np.mean(axis=(0, 1), keepdims=True)
        image_np = mean + (image_np - mean) * contrast

        mult_std = float(cfg.get("multiplicative_noise_std", 0.06))
        gauss_std = float(cfg.get("gaussian_noise_std", 0.025)) * 255.0

        if mult_std > 0:
            mult = np.random.normal(1.0, mult_std, size=image_np.shape[:2] + (1,)).astype(np.float32)
            image_np = image_np * mult

        if gauss_std > 0:
            noise = np.random.normal(0.0, gauss_std, size=image_np.shape).astype(np.float32)
            image_np = image_np + noise

        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        return (
            Image.fromarray(image_np),
            {
                "noise_contrast": True,
                "contrast": float(contrast),
                "multiplicative_noise_std": float(mult_std),
                "gaussian_noise_std": float(gauss_std / 255.0),
            },
        )

    def _apply_training_augmentation(
        self,
        image: Image.Image,
        polygons: List[Dict],
    ) -> Tuple[Image.Image, List[Dict], Dict]:
        meta = {}

        image, polygons, scale_meta = self._apply_nonuniform_scale(image, polygons)
        meta.update(scale_meta)

        image, polygons, elastic_meta = self._apply_elastic_distortion(image, polygons)
        meta.update(elastic_meta)

        image, blur_meta = self._apply_blur(image)
        meta.update(blur_meta)

        image, noise_meta = self._apply_noise_and_contrast(image)
        meta.update(noise_meta)

        return image, polygons, meta

    def apply_augmentation(
        self,
        image: Image.Image,
        polygons: List[Dict],
    ) -> Tuple[Image.Image, List[Dict], Dict]:
        """Apply only the configured degradation, before resize/normalization.

        This public stage lets E2E evaluation degrade the full page and its GT
        geometry identically to detection validation, while the OCR recognizer
        still receives crops made exclusively from detector-predicted boxes.
        """
        image = image.convert("RGB")
        if not self.augment:
            return image, polygons, {"enabled": False}

        image, polygons, meta = self._apply_training_augmentation(image, polygons)
        meta["enabled"] = True
        return image, polygons, meta

    def __call__(
        self,
        image: Image.Image,
        polygons: List[Dict],
    ) -> Tuple[torch.Tensor, List[Dict], Dict]:
        image, polygons, aug_meta = self.apply_augmentation(image, polygons)

        orig_w, orig_h = image.size

        if self.keep_aspect_ratio:
            scale = min(
                self.target_width / float(orig_w),
                self.target_height / float(orig_h),
            )

            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))

            new_w = max(1, min(new_w, self.target_width))
            new_h = max(1, min(new_h, self.target_height))

            resized = image.resize((new_w, new_h), Image.BILINEAR)

            canvas = Image.new(
                "RGB",
                (self.target_width, self.target_height),
                color=(255, 255, 255),
            )
            canvas.paste(resized, (0, 0))

            scale_x = new_w / float(orig_w)
            scale_y = new_h / float(orig_h)
            pad_x = 0
            pad_y = 0

            image = canvas
        else:
            image = image.resize((self.target_width, self.target_height), Image.BILINEAR)

            scale_x = self.target_width / float(orig_w)
            scale_y = self.target_height / float(orig_h)
            pad_x = 0
            pad_y = 0

        scaled_polygons = []

        for poly in polygons:
            points = poly["points"]

            new_points = []
            for x, y in points:
                new_x = float(x) * scale_x + pad_x
                new_y = float(y) * scale_y + pad_y
                new_points.append([new_x, new_y])

            xs = [p[0] for p in new_points]
            ys = [p[1] for p in new_points]

            scaled_polygons.append(
                {
                    "label": poly.get("label", "manchu"),
                    "points": new_points,
                    "bbox": [
                        float(min(xs)),
                        float(min(ys)),
                        float(max(xs)),
                        float(max(ys)),
                    ],
                }
            )

        tensor = self.to_tensor(image)
        tensor = self.normalize(tensor)

        meta = {
            "orig_width": orig_w,
            "orig_height": orig_h,
            "resized_width": int(round(orig_w * scale_x)),
            "resized_height": int(round(orig_h * scale_y)),
            "target_width": self.target_width,
            "target_height": self.target_height,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "augmentation": aug_meta,
        }

        return tensor, scaled_polygons, meta


def build_det_transform(
    target_height: int = 1056,
    target_width: int = 768,
    keep_aspect_ratio: bool = True,
    augment: bool = False,
    augmentation_cfg: Dict | None = None,
):
    return DetResizePad(
        target_height=target_height,
        target_width=target_width,
        keep_aspect_ratio=keep_aspect_ratio,
        augment=augment,
        augmentation_cfg=augmentation_cfg,
    )
