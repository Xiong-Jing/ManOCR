import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Masked Dice loss for probability or binary maps."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pred = pred.float()
        gt = gt.to(pred.device).float()

        if mask is None:
            mask = torch.ones_like(gt)
        else:
            mask = mask.to(pred.device).float()

        pred = pred * mask
        gt = gt * mask

        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() + self.eps
        return 1.0 - 2.0 * intersection / union
