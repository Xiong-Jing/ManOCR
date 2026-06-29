from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthographicTransitionLoss(nn.Module):
    """
    Orthographic Transition-Penalty Loss for Manchu romanized transcription.

    Model logits:
        [B, T, C]

    C includes CTC blank.
    transition_matrix:
        [K, K], where K = C - 1 if blank_idx = 0.

    This implementation:
        - removes the CTC blank class
        - normalizes non-blank character probabilities
        - computes transition likelihood between adjacent time steps
        - applies confidence weighting to reduce the impact of blank-dominated steps
    """

    def __init__(
        self,
        transition_matrix_path: str | Path,
        blank_idx: int = 0,
        eps: float = 1e-8,
        normalize_char_probs: bool = True,
        confidence_weighting: bool = True,
        confidence_power: float = 1.0,
        min_confidence: float = 0.0,
    ):
        super().__init__()

        transition_matrix_path = Path(transition_matrix_path)

        if not transition_matrix_path.exists():
            raise FileNotFoundError(f"Transition matrix not found: {transition_matrix_path}")

        matrix = np.load(transition_matrix_path)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Transition matrix must be square, got shape: {matrix.shape}"
            )

        matrix = matrix.astype("float32")
        matrix = np.clip(matrix, eps, None)

        # Normalize again for safety.
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), eps)

        self.register_buffer("transition_matrix", torch.from_numpy(matrix))

        self.blank_idx = blank_idx
        self.eps = eps
        self.normalize_char_probs = normalize_char_probs
        self.confidence_weighting = confidence_weighting
        self.confidence_power = float(confidence_power)
        self.min_confidence = float(min_confidence)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, C]

        Returns:
            scalar loss
        """
        if logits.dim() != 3:
            raise ValueError(f"logits should be [B, T, C], got {tuple(logits.shape)}")

        # Important: compute Lortho in float32 for numerical stability.
        logits = logits.float()

        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits detected before OrthographicTransitionLoss.")

        # Avoid extreme logits causing unstable softmax.
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        batch_size, time_steps, num_classes = logits.shape

        if time_steps < 2:
            return logits.new_tensor(0.0)

        num_chars = num_classes - 1
        matrix_size = self.transition_matrix.shape[0]

        if num_chars != matrix_size:
            raise ValueError(
                f"Transition matrix size mismatch. "
                f"Model has {num_chars} non-blank chars, "
                f"but matrix shape is {tuple(self.transition_matrix.shape)}"
            )

        if self.blank_idx != 0:
            raise NotImplementedError(
                "Current project assumes CTC blank_idx = 0. "
                "Please keep blank at index 0."
            )

        transition_matrix = self.transition_matrix.to(
            device=logits.device,
            dtype=logits.dtype,
        ).clamp_min(self.eps)

        transition_matrix = transition_matrix / transition_matrix.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)

        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        char_probs = probs[:, :, 1:]

        if self.normalize_char_probs:
            char_probs = char_probs / char_probs.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)

        p_t = char_probs[:, :-1, :]
        p_next = char_probs[:, 1:, :]

        transition_scores = torch.einsum(
            "bti,ij,btj->bt",
            p_t,
            transition_matrix,
            p_next,
        )

        transition_scores = torch.nan_to_num(
            transition_scores,
            nan=self.eps,
            posinf=1.0,
            neginf=self.eps,
        ).clamp_min(self.eps)

        raw_loss = -torch.log(transition_scores)

        # Prevent one unstable transition from dominating the whole loss.
        raw_loss = torch.clamp(raw_loss, max=20.0)

        if self.confidence_weighting:
            non_blank_conf_t = 1.0 - probs[:, :-1, self.blank_idx]
            non_blank_conf_next = 1.0 - probs[:, 1:, self.blank_idx]
            weights = (non_blank_conf_t * non_blank_conf_next).detach()
            weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)

            if self.min_confidence > 0:
                weights = torch.where(
                    weights >= self.min_confidence,
                    weights,
                    torch.zeros_like(weights),
                )

            if self.confidence_power != 1.0:
                weights = weights.clamp_min(0.0).pow(self.confidence_power)

            loss = (raw_loss * weights).sum() / weights.sum().clamp_min(1.0)
        else:
            loss = raw_loss.mean()

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite OrthographicTransitionLoss detected.")

        return loss
