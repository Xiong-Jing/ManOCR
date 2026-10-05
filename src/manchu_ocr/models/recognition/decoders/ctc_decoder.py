from __future__ import annotations

from typing import Dict, List

import torch


class CTCGreedyDecoder:
    """Greedy CTC decoder for index tensors or logits."""

    def __init__(self, idx_to_char: Dict[int, str], blank_idx: int = 0):
        self.idx_to_char = idx_to_char
        self.blank_idx = blank_idx

    def decode_indices(self, indices: torch.Tensor, raw: bool = False) -> List[str]:
        if indices.dim() != 2:
            raise ValueError(f"indices should be [B, T], got {tuple(indices.shape)}")

        results = []
        for seq in indices.detach().cpu().tolist():
            chars = []
            prev = None
            for idx in seq:
                if raw:
                    chars.append(self.idx_to_char.get(int(idx), ""))
                elif idx != self.blank_idx and idx != prev:
                    chars.append(self.idx_to_char.get(int(idx), ""))
                prev = idx
            results.append("".join(chars))
        return results

    def decode_logits(self, logits: torch.Tensor) -> List[str]:
        if logits.dim() != 3:
            raise ValueError(f"logits should be [B, T, C], got {tuple(logits.shape)}")
        return self.decode_indices(logits.argmax(dim=-1), raw=False)
