from dataclasses import dataclass
import random

from PIL import Image


@dataclass
class ManchuAspectJitter:
    """Lightweight aspect-ratio jitter for vertical word crops."""

    min_scale: float = 0.9
    max_scale: float = 1.1

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = random.uniform(self.min_scale, self.max_scale)
        new_width = max(1, int(round(width * scale)))
        return image.resize((new_width, height), Image.BILINEAR)
