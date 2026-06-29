from pathlib import Path
from typing import Sequence

import cv2

from manchu_ocr.utils.visualizer import imread_unicode, imwrite_unicode


def draw_recognition_results(
    image_path: str | Path,
    lines: Sequence[str],
    output_path: str | Path,
) -> None:
    image = imread_unicode(image_path)
    y = 24
    for line in lines:
        cv2.putText(image, str(line), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 28
    imwrite_unicode(output_path, image)
