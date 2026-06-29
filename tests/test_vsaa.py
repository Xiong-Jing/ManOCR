import torch

from manchu_ocr.models.detection.modules.vsaa import VSAA


def test_vsaa_preserves_shape():
    module = VSAA(channels=32, vertical_kernel=5, horizontal_kernel=3)
    x = torch.randn(2, 32, 16, 12)
    y = module(x)

    assert y.shape == x.shape
