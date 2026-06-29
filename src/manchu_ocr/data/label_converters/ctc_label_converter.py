from pathlib import Path
from typing import List, Sequence, Tuple

import torch


class CTCLabelConverter:
    """
    Convert between text labels and CTC target indices.

    Index design:
        0: CTC blank
        1...N: real characters from charset.txt
    """

    def __init__(self, charset_path: str | Path):
        charset_path = Path(charset_path)

        if not charset_path.exists():
            raise FileNotFoundError(f"Charset file not found: {charset_path}")

        with charset_path.open("r", encoding="utf-8") as f:
            chars = [line.rstrip("\n") for line in f if line.rstrip("\n") != ""]

        if len(chars) == 0:
            raise ValueError(f"Empty charset file: {charset_path}")

        if len(set(chars)) != len(chars):
            raise ValueError(f"Duplicated characters found in charset: {charset_path}")

        self.blank_idx = 0
        self.chars = chars
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(chars)}
        self.idx_to_char[self.blank_idx] = ""

        self.num_classes = len(chars) + 1

    def encode(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of text labels for torch.nn.CTCLoss.

        Args:
            texts: list of strings.

        Returns:
            targets: 1D LongTensor, concatenated encoded labels.
            target_lengths: 1D LongTensor, length of each label.
        """
        encoded = []
        lengths = []

        for text in texts:
            indices = []

            for char in text:
                if char not in self.char_to_idx:
                    raise KeyError(
                        f"Character {repr(char)} not found in charset. "
                        f"Text: {repr(text)}"
                    )
                indices.append(self.char_to_idx[char])

            encoded.extend(indices)
            lengths.append(len(indices))

        targets = torch.tensor(encoded, dtype=torch.long)
        target_lengths = torch.tensor(lengths, dtype=torch.long)

        return targets, target_lengths

    def decode(
        self,
        preds: torch.Tensor,
        raw: bool = False,
    ) -> List[str]:
        """
        Decode prediction indices.

        Args:
            preds:
                Shape [B, T] or [T, B].
                If [T, B], transpose before passing or set outside.
            raw:
                If True, keep repeated characters and blanks.
                If False, perform standard CTC decoding.

        Returns:
            List of decoded strings.
        """
        if preds.dim() != 2:
            raise ValueError(f"preds should be 2D, got shape {tuple(preds.shape)}")

        results = []

        for seq in preds:
            chars = []
            prev_idx = None

            for idx in seq.tolist():
                if raw:
                    chars.append(self.idx_to_char.get(idx, ""))
                    continue

                if idx != self.blank_idx and idx != prev_idx:
                    chars.append(self.idx_to_char.get(idx, ""))

                prev_idx = idx

            results.append("".join(chars))

        return results

    def decode_logits(self, logits: torch.Tensor) -> List[str]:
        """
        Decode logits from model.

        Args:
            logits: Tensor with shape [B, T, C].

        Returns:
            List of decoded strings.
        """
        if logits.dim() != 3:
            raise ValueError(f"logits should be [B, T, C], got {tuple(logits.shape)}")

        pred_indices = logits.argmax(dim=-1)
        return self.decode(pred_indices, raw=False)
