import torch
import torch.nn as nn
import torch.nn.functional as F


class NRTRLoss(nn.Module):
    """Token-level autoregressive cross-entropy for the auxiliary NRTR head."""

    def __init__(self, pad_idx: int = 0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.pad_idx = int(pad_idx)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError(f"logits must be [B, L, V], got {tuple(logits.shape)}")
        if targets.dim() != 2:
            raise ValueError(f"targets must be [B, L], got {tuple(targets.shape)}")
        if logits.shape[:2] != targets.shape:
            raise ValueError(
                "NRTR logits/targets shape mismatch: "
                f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
            )

        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
            label_smoothing=self.label_smoothing,
        )
