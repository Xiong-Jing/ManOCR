from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceAlignmentLoss(nn.Module):
    """
    Lightweight auxiliary loss for CTC recognition.

    CTC provides weak sequence-level supervision and may plateau with many
    one-character word errors. This loss places each target character at a
    uniformly spaced position along the output sequence and applies CE there.

    It is intentionally auxiliary: CTC remains the main loss.
    """

    def __init__(
        self,
        blank_idx: int = 0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.blank_idx = blank_idx
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError(f"logits should be [B, T, C], got {tuple(logits.shape)}")

        batch_size, time_steps, num_classes = logits.shape

        losses = []
        offset = 0

        for batch_idx in range(batch_size):
            length = int(target_lengths[batch_idx].item())

            if length <= 0:
                continue

            sample_targets = targets[offset: offset + length]
            offset += length

            if sample_targets.numel() == 0:
                continue

            # Put targets at segment centers. This gives CTC a monotonic
            # alignment hint without forcing every frame to be non-blank.
            positions = (
                (torch.arange(length, device=logits.device, dtype=torch.float32) + 0.5)
                * float(time_steps)
                / float(length)
            )
            positions = positions.floor().long().clamp(0, time_steps - 1)

            sample_logits = logits[batch_idx, positions, :]

            loss = F.cross_entropy(
                sample_logits,
                sample_targets.to(logits.device),
                label_smoothing=self.label_smoothing,
            )
            losses.append(loss)

        if not losses:
            return logits.sum() * 0.0

        return torch.stack(losses).mean()
