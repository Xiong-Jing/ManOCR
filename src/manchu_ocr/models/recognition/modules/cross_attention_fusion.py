import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Fuse global SVTR features with fine-grained DAB features.

    Query: global SVTR sequence
    Key/Value: DAB detail sequence
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 6,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        attn_residual_scale: float = 0.1,
        ffn_residual_scale: float = 0.1,
        max_residual_scale: float = 0.15,
        gate_bias: float = -2.0,
    ):
        super().__init__()

        if max_residual_scale <= 0:
            raise ValueError("max_residual_scale should be positive.")

        self.max_residual_scale = float(max_residual_scale)

        self.query_norm = nn.LayerNorm(embed_dim)
        self.detail_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.gate_norm = nn.LayerNorm(embed_dim * 2)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        hidden_dim = int(embed_dim * mlp_ratio)

        self.detail_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.detail_gate = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.zeros_(self.detail_gate.weight)
        nn.init.constant_(self.detail_gate.bias, float(gate_bias))

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Start close to the baseline recognizer and let training decide how much
        # fine-detail information should affect the main SVTR sequence.
        self.attn_residual_scale = nn.Parameter(
            self._scale_to_logit(attn_residual_scale)
        )
        self.ffn_residual_scale = nn.Parameter(
            self._scale_to_logit(ffn_residual_scale)
        )

    def _scale_to_logit(self, value: float) -> torch.Tensor:
        eps = 1e-4
        ratio = float(value) / self.max_residual_scale
        ratio = min(max(ratio, eps), 1.0 - eps)
        return torch.logit(torch.tensor(ratio, dtype=torch.float32))

    def _bounded_scale(self, raw_scale: torch.Tensor) -> torch.Tensor:
        return self.max_residual_scale * torch.sigmoid(raw_scale)

    def forward(
        self,
        global_feat: torch.Tensor,
        detail_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            global_feat: [B, T, C]
            detail_feat: [B, T, C]

        Returns:
            fused feature: [B, T, C]
        """
        if global_feat.shape[1] != detail_feat.shape[1]:
            detail_feat = self._resize_sequence(detail_feat, global_feat.shape[1])

        query = self.query_norm(global_feat)
        detail = self.detail_norm(detail_feat)

        attn_out, _ = self.attn(
            query=query,
            key=detail,
            value=detail,
            need_weights=False,
        )
        detail_out = self.detail_proj(detail)
        detail_gate = torch.sigmoid(
            self.detail_gate(
                self.gate_norm(torch.cat([query, detail], dim=-1))
            )
        )

        attn_scale = self._bounded_scale(self.attn_residual_scale)
        ffn_scale = self._bounded_scale(self.ffn_residual_scale)

        x = global_feat + attn_scale * detail_gate * self.dropout(attn_out + detail_out)
        x = x + ffn_scale * self.ffn(self.ffn_norm(x))

        return x

    @staticmethod
    def _resize_sequence(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Resize sequence length if global branch and detail branch differ.
        """
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        x = x.transpose(1, 2).contiguous()
        return x
