from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DBLoss(nn.Module):
    """
    DBNet loss.

    total_loss = prob_loss + alpha * binary_loss + beta * thresh_loss
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 10.0,
        gamma_as: float = 0.0,
        negative_ratio: float = 3.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma_as = gamma_as
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pred_prob = preds["prob_map"].float()
        pred_thresh = preds["thresh_map"].float()
        pred_binary = preds["binary_map"].float()

        gt_prob = batch["prob_map"].to(pred_prob.device).float()
        gt_thresh = batch["thresh_map"].to(pred_prob.device).float()
        gt_thresh_mask = batch["thresh_mask"].to(pred_prob.device).float()
        gt_training_mask = batch["training_mask"].to(pred_prob.device).float()

        prob_loss = self._balanced_bce_loss(
            pred=pred_prob,
            gt=gt_prob,
            mask=gt_training_mask,
        )

        thresh_loss = self._masked_l1_loss(
            pred=pred_thresh,
            gt=gt_thresh,
            mask=gt_thresh_mask,
        )

        binary_loss = self._dice_loss(
            pred=pred_binary,
            gt=gt_prob,
            mask=gt_training_mask,
        )

        as_loss = pred_prob.sum() * 0.0

        if self.gamma_as > 0:
            if "as_prob_map" not in preds:
                raise KeyError(
                    "DBLoss received gamma_as > 0, but model outputs do not include "
                    "'as_prob_map'. Enable model.head.as_auxiliary_head for AS runs."
                )
            if "as_prob_map" not in batch:
                raise KeyError(
                    "DBLoss received gamma_as > 0, but batch does not include "
                    "'as_prob_map'. Enable data.label_generator.use_asymmetric_shrink "
                    "and make sure DetCollate keeps the AS target."
                )

            pred_as_prob = preds["as_prob_map"].float()
            gt_as_prob = batch["as_prob_map"].to(pred_prob.device).float()
            as_loss = self._positive_bce_loss(
                pred=pred_as_prob,
                gt=gt_as_prob,
                mask=gt_training_mask,
            )

        loss = (
            prob_loss
            + self.alpha * binary_loss
            + self.beta * thresh_loss
            + self.gamma_as * as_loss
        )

        return {
            "loss": loss,
            "prob_loss": prob_loss.detach(),
            "binary_loss": binary_loss.detach(),
            "thresh_loss": thresh_loss.detach(),
            "as_loss": as_loss.detach(),
        }

    def _balanced_bce_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.clamp(self.eps, 1.0 - self.eps)

        bce = F.binary_cross_entropy(
            pred,
            gt,
            reduction="none",
        )

        positive = (gt > 0.5) & (mask > 0.5)
        negative = (gt <= 0.5) & (mask > 0.5)

        positive_loss = bce[positive]
        negative_loss = bce[negative]

        pos_count = positive_loss.numel()
        neg_count = negative_loss.numel()

        if pos_count == 0:
            if neg_count == 0:
                return pred.sum() * 0.0
            return negative_loss.mean()

        max_neg_count = int(pos_count * self.negative_ratio)
        selected_neg_count = min(neg_count, max_neg_count)

        if selected_neg_count > 0:
            negative_loss, _ = torch.topk(negative_loss, selected_neg_count)
            loss = (
                positive_loss.sum() + negative_loss.sum()
            ) / (pos_count + selected_neg_count + self.eps)
        else:
            loss = positive_loss.mean()

        return loss

    def _masked_l1_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.abs(pred - gt) * mask
        return loss.sum() / (mask.sum() + self.eps)

    def _dice_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred * mask
        gt = gt * mask

        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() + self.eps

        loss = 1.0 - 2.0 * intersection / union
        return loss

    def _positive_bce_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary AS supervision.

        It only encourages responses on AS positive pixels and does not
        penalize predictions outside the auxiliary AS target. The standard DB
        prob/binary losses remain responsible for negative supervision.
        """
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        positive = (gt > 0.5) & (mask > 0.5)

        if not bool(positive.any()):
            return pred.sum() * 0.0

        return (-torch.log(pred[positive])).mean()
