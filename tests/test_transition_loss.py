from pathlib import Path

import numpy as np
import torch

from manchu_ocr.losses.orthographic_transition_loss import OrthographicTransitionLoss


def test_orthographic_transition_loss_is_finite(tmp_path: Path):
    matrix_path = tmp_path / "transition.npy"
    np.save(matrix_path, np.ones((2, 2), dtype=np.float32) / 2.0)

    criterion = OrthographicTransitionLoss(matrix_path)
    logits = torch.randn(2, 4, 3)
    loss = criterion(logits)

    assert torch.isfinite(loss)
