from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


class CTCBeamSearchDecoder:
    """Small prefix beam-search decoder for CTC logits."""

    def __init__(self, idx_to_char: Dict[int, str], blank_idx: int = 0, beam_size: int = 5):
        self.idx_to_char = idx_to_char
        self.blank_idx = blank_idx
        self.beam_size = beam_size

    def decode_logits(self, logits: torch.Tensor) -> List[str]:
        if logits.dim() != 3:
            raise ValueError(f"logits should be [B, T, C], got {tuple(logits.shape)}")

        log_probs = F.log_softmax(logits.float(), dim=-1).detach().cpu()
        return [self._decode_single(item) for item in log_probs]

    def _decode_single(self, log_probs: torch.Tensor) -> str:
        beams: Dict[Tuple[int, ...], float] = {(): 0.0}

        for step in log_probs:
            next_beams: Dict[Tuple[int, ...], float] = defaultdict(lambda: float("-inf"))
            top_scores, top_indices = torch.topk(step, k=min(self.beam_size, step.numel()))

            for prefix, score in beams.items():
                for token_score, token_idx in zip(top_scores.tolist(), top_indices.tolist()):
                    token_idx = int(token_idx)
                    new_score = score + float(token_score)
                    if token_idx == self.blank_idx:
                        new_prefix = prefix
                    elif prefix and token_idx == prefix[-1]:
                        new_prefix = prefix
                    else:
                        new_prefix = prefix + (token_idx,)

                    next_beams[new_prefix] = max(next_beams[new_prefix], new_score)

            beams = dict(sorted(next_beams.items(), key=lambda x: x[1], reverse=True)[: self.beam_size])

        best = max(beams.items(), key=lambda x: x[1])[0]
        return "".join(self.idx_to_char.get(idx, "") for idx in best)
