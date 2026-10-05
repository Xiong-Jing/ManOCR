import torch
import torch.nn as nn


class SVTRBackbone(nn.Module):
    """
    Minimal Transformer encoder backbone for SVTR baseline.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]

        Returns:
            features: [B, T, C]
        """
        b, t, c = x.shape

        if t > self.pos_embed.shape[1]:
            raise ValueError(
                f"Sequence length {t} exceeds max_seq_len {self.pos_embed.shape[1]}"
            )

        x = x + self.pos_embed[:, :t, :]
        x = self.pos_drop(x)

        x = self.encoder(x)
        x = self.norm(x)

        return x
