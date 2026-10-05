import torch
import torch.nn as nn


class DifferentiableBinarization(nn.Module):
    """DBNet differentiable step function."""

    def __init__(self, k: int = 50):
        super().__init__()
        self.k = k

    def forward(self, prob_map: torch.Tensor, thresh_map: torch.Tensor) -> torch.Tensor:
        return torch.reciprocal(1.0 + torch.exp(-self.k * (prob_map - thresh_map)))
