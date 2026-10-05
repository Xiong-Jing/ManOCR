import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedBCELoss(nn.Module):
    """Balanced BCE with hard negative mining, matching DBNet probability loss."""

    def __init__(self, negative_ratio: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = pred.float().clamp(self.eps, 1.0 - self.eps)
        gt = gt.to(pred.device).float()
        mask = mask.to(pred.device).float()

        loss = F.binary_cross_entropy(pred, gt, reduction="none")
        positive = (gt > 0.5) & (mask > 0.5)
        negative = (gt <= 0.5) & (mask > 0.5)

        pos_loss = loss[positive]
        neg_loss = loss[negative]

        if pos_loss.numel() == 0:
            if neg_loss.numel() == 0:
                return pred.sum() * 0.0
            return neg_loss.mean()

        neg_count = min(neg_loss.numel(), int(pos_loss.numel() * self.negative_ratio))
        if neg_count > 0:
            neg_loss, _ = torch.topk(neg_loss, neg_count)
            return (pos_loss.sum() + neg_loss.sum()) / (pos_loss.numel() + neg_count + self.eps)

        return pos_loss.mean()
