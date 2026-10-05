from pathlib import Path
from typing import Sequence

from manchu_ocr.utils.visualizer import draw_polygons, imread_unicode, imwrite_unicode


def draw_detection_boxes(
    image_path: str | Path,
    boxes: Sequence[Sequence[Sequence[float]]],
    output_path: str | Path,
) -> None:
    image = imread_unicode(image_path)
    vis = draw_polygons(image, boxes, color=(0, 0, 255), thickness=2)
    imwrite_unicode(output_path, vis)
