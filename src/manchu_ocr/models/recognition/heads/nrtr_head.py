from __future__ import annotations

import torch
import torch.nn as nn


class NRTRDecoderHead(nn.Module):
    """Autoregressive Transformer decoder used as an auxiliary NRTR head.

    Token indices preserve the project's CTC character indices:

    - ``0`` is both the CTC blank and the NRTR padding index;
    - ``1 ... num_ctc_classes - 1`` are real characters;
    - ``num_ctc_classes`` is BOS;
    - ``num_ctc_classes + 1`` is EOS.

    The head is trained with teacher forcing. Inference remains on the CTC
    branch, so adding this auxiliary decoder does not change recognition
    decoding or evaluation metrics.
    """

    def __init__(
        self,
        memory_dim: int,
        num_ctc_classes: int,
        decoder_dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
        max_text_length: int = 128,
    ) -> None:
        super().__init__()

        if num_ctc_classes < 2:
            raise ValueError("num_ctc_classes must include blank and at least one character.")
        if max_text_length < 1:
            raise ValueError("max_text_length must be positive.")
        if decoder_dim % num_heads != 0:
            raise ValueError("decoder_dim must be divisible by num_heads.")

        self.num_ctc_classes = int(num_ctc_classes)
        self.pad_idx = 0
        self.bos_idx = self.num_ctc_classes
        self.eos_idx = self.num_ctc_classes + 1
        self.vocab_size = self.num_ctc_classes + 2
        self.max_text_length = int(max_text_length)

        self.memory_projection = (
            nn.Identity()
            if int(memory_dim) == int(decoder_dim)
            else nn.Linear(int(memory_dim), int(decoder_dim))
        )
        self.memory_norm = nn.LayerNorm(int(decoder_dim))
        self.token_embedding = nn.Embedding(
            self.vocab_size,
            int(decoder_dim),
            padding_idx=self.pad_idx,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_text_length + 1, int(decoder_dim))
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=int(decoder_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(int(decoder_dim)),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(decoder_dim), self.vocab_size)

    def build_teacher_forcing_sequences(
        self,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert packed CTC targets into NRTR decoder inputs and targets."""

        if targets.dim() != 1:
            raise ValueError(f"targets must be 1D, got {tuple(targets.shape)}")
        if target_lengths.dim() != 1:
            raise ValueError(
                f"target_lengths must be 1D, got {tuple(target_lengths.shape)}"
            )

        lengths = [int(length) for length in target_lengths.detach().cpu().tolist()]
        if any(length < 0 for length in lengths):
            raise ValueError("target_lengths cannot contain negative values.")
        if sum(lengths) != int(targets.numel()):
            raise ValueError(
                "Packed targets do not match target_lengths: "
                f"num_targets={targets.numel()}, sum_lengths={sum(lengths)}"
            )

        max_length = max(lengths, default=0)
        if max_length > self.max_text_length:
            raise ValueError(
                f"Target length {max_length} exceeds NRTR max_text_length "
                f"{self.max_text_length}."
            )
        if targets.numel() > 0:
            min_target = int(targets.min().item())
            max_target = int(targets.max().item())
            if min_target <= self.pad_idx or max_target >= self.num_ctc_classes:
                raise ValueError(
                    "NRTR character targets must use the CTC character range "
                    f"[1, {self.num_ctc_classes - 1}], got "
                    f"[{min_target}, {max_target}]."
                )

        batch_size = len(lengths)
        sequence_length = max_length + 1
        decoder_inputs = targets.new_full(
            (batch_size, sequence_length),
            fill_value=self.pad_idx,
        )
        decoder_targets = targets.new_full(
            (batch_size, sequence_length),
            fill_value=self.pad_idx,
        )
        decoder_inputs[:, 0] = self.bos_idx

        offset = 0
        for batch_idx, length in enumerate(lengths):
            if length > 0:
                sample_targets = targets[offset : offset + length]
                decoder_inputs[batch_idx, 1 : length + 1] = sample_targets
                decoder_targets[batch_idx, :length] = sample_targets
            decoder_targets[batch_idx, length] = self.eos_idx
            offset += length

        padding_mask = decoder_inputs.eq(self.pad_idx)
        return decoder_inputs, decoder_targets, padding_mask

    def forward(
        self,
        memory: torch.Tensor,
        decoder_inputs: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory.dim() != 3:
            raise ValueError(f"memory must be [B, T, C], got {tuple(memory.shape)}")
        if decoder_inputs.dim() != 2:
            raise ValueError(
                f"decoder_inputs must be [B, L], got {tuple(decoder_inputs.shape)}"
            )
        if decoder_inputs.shape[1] > self.position_embedding.shape[1]:
            raise ValueError(
                f"Decoder length {decoder_inputs.shape[1]} exceeds positional "
                f"capacity {self.position_embedding.shape[1]}."
            )

        memory = self.memory_norm(self.memory_projection(memory))
        target = self.token_embedding(decoder_inputs)
        target = target + self.position_embedding[:, : target.shape[1]]
        target = self.dropout(target)

        target_length = target.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                target_length,
                target_length,
                dtype=torch.bool,
                device=target.device,
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        return self.classifier(decoded)
