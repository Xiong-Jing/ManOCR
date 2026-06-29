import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(nn.Module):
    """
    Wrapper for torch.nn.CTCLoss.

    Model logits:
        [B, T, C]

    CTCLoss expects:
        log_probs [T, B, C]
    """

    def __init__(
        self,
        blank_idx: int = 0,
        zero_infinity: bool = True,
    ):
        super().__init__()

        self.blank_idx = blank_idx
        self.loss_fn = nn.CTCLoss(
            blank=blank_idx,
            reduction="mean",
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError(f"logits should be [B, T, C], got {tuple(logits.shape)}")

        batch_size, time_steps, _ = logits.shape

        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2).contiguous()  # [T, B, C]

        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=time_steps,
            dtype=torch.long,
            device=logits.device,
        )

        loss = self.loss_fn(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        return loss
