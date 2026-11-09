"""Paper Table 3 augmentation with a shared forward warp for pixels and vertices."""
import random

import cv2
import numpy as np
from PIL import Image


def sample_field(field, x, y):
    return cv2.remap(field, np.asarray(x, np.float32), np.asarray(y, np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def elastic_warp(image, polygons, dx, dy):
    from manchu_ocr.data.transforms.det_transforms import _update_polygon_points
    width, height = image.size
    gx, gy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    # Invert destination = source + displacement(source), not an inconsistent
    # destination - displacement(destination) approximation for the image alone.
    sx, sy = gx.copy(), gy.copy()
    for _ in range(20):
        next_x, next_y = gx - sample_field(dx, sx, sy), gy - sample_field(dy, sx, sy)
        error = max(float(np.max(np.abs(next_x - sx))), float(np.max(np.abs(next_y - sy))))
        sx, sy = next_x, next_y
        if error < 0.01:
            break
    warped = cv2.remap(np.asarray(image), sx, sy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    transformed = []
    for poly in polygons:
        points = np.asarray(poly["points"], dtype=np.float32)
        x, y = points[:, 0:1], points[:, 1:2]
        offset = np.column_stack((sample_field(dx, x, y).ravel(), sample_field(dy, x, y).ravel()))
        transformed.append(_update_polygon_points(poly, points + offset, width, height))
    return Image.fromarray(warped), transformed


def paper_augmentation(image, polygons):
    from manchu_ocr.data.transforms.det_transforms import _update_polygon_points
    width, height = image.size
    brightness, contrast = random.uniform(0.85, 1.15), random.uniform(0.8, 1.2)
    values = np.asarray(image, dtype=np.float32)
    mean = values.mean(axis=(0, 1), keepdims=True)
    values = np.clip((mean + contrast * (values - mean)) * brightness, 0, 255).astype(np.uint8)
    sh, sv = random.uniform(0.9, 1.1), random.uniform(0.85, 1.15)
    affine = np.array([[sh, 0, width * (1 - sh) / 2], [0, sv, height * (1 - sv) / 2]], np.float32)
    values = cv2.warpAffine(values, affine, (width, height), borderValue=(255, 255, 255))
    scaled = []
    for poly in polygons:
        points = np.asarray(poly["points"], np.float32)
        mapped = np.column_stack((points, np.ones(len(points)))) @ affine.T
        scaled.append(_update_polygon_points(poly, mapped, width, height))
    ax, ay = random.uniform(0, 0.01) * width, random.uniform(0, 0.03) * height
    sigma = random.uniform(0.8, 1.2)
    fields = []
    for amplitude in (ax, ay):
        grid = np.random.uniform(-amplitude, amplitude, (8, 8)).astype(np.float32)
        grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=sigma)
        fields.append(cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR))
    image, polygons = elastic_warp(Image.fromarray(values), scaled, *fields)
    return image, polygons, {"profile": "paper_table3", "brightness": brightness, "contrast": contrast,
                             "scale_x": sh, "scale_y": sv, "alpha_x": ax, "alpha_y": ay,
                             "grid_size": 8, "sigma_grid_intervals": sigma}
