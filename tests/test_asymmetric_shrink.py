import numpy as np

from manchu_ocr.data.label_generators.asymmetric_shrink import shrink_vertical_box


def test_asymmetric_shrink_keeps_polygon_valid():
    points = np.asarray([[0, 0], [10, 0], [10, 100], [0, 100]], dtype=np.float32)
    shrinked = shrink_vertical_box(points, shrink_ratio_x=0.5, shrink_ratio_y=0.9)

    assert shrinked is not None
    assert shrinked.shape == points.shape
    assert shrinked[:, 0].min() > points[:, 0].min()
    assert shrinked[:, 0].max() < points[:, 0].max()
