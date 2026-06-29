from dataclasses import dataclass
import math
import random

from PIL import Image


@dataclass
class VerticalElasticTransform:
    """Simple vertical sinusoidal warp for robustness experiments."""

    amplitude: float = 2.0
    period: float = 48.0
    probability: float = 0.5

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image

        image = image.convert("RGB")
        width, height = image.size
        output = Image.new("RGB", (width, height), color=(255, 255, 255))

        phase = random.uniform(0.0, 2.0 * math.pi)
        for y in range(height):
            offset = int(round(self.amplitude * math.sin(2.0 * math.pi * y / self.period + phase)))
            row = image.crop((0, y, width, y + 1))
            output.paste(row, (offset, y))

        return output
