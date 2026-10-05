import torch
import torch.nn as nn


class CTCHead(nn.Module):
    """
    Linear classification head for CTC.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]

        Returns:
            logits: [B, T, num_classes]
        """
        return self.fc(x)
