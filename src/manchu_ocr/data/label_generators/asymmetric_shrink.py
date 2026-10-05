import numpy as np

from manchu_ocr.data.label_generators.db_label_generator import asymmetric_shrink_polygon


def shrink_vertical_box(
    points: np.ndarray,
    shrink_ratio_x: float = 0.65,
    shrink_ratio_y: float = 0.90,
) -> np.ndarray | None:
    """Alias used by experiments for Manchu vertical text asymmetric shrink."""
    return asymmetric_shrink_polygon(
        points=np.asarray(points, dtype=np.float32),
        shrink_ratio_x=shrink_ratio_x,
        shrink_ratio_y=shrink_ratio_y,
    )


__all__ = ["asymmetric_shrink_polygon", "shrink_vertical_box"]
