from typing import Literal

import torch
import torch.nn as nn

from manchu_ocr.models.recognition.backbones.patch_embed import SequencePatchEmbed
from manchu_ocr.models.recognition.backbones.svtr_backbone import SVTRBackbone
from manchu_ocr.models.recognition.branches.diacritic_aware_branch import DiacriticAwareBranch
from manchu_ocr.models.recognition.heads.ctc_head import CTCHead
from manchu_ocr.models.recognition.modules.cross_attention_fusion import CrossAttentionFusion


class SVTRRecognizer(nn.Module):
    """
    SVTR recognizer for Manchu word recognition.

    Supported modes:
        - SVTR baseline
        - SVTR + Diacritic-Aware Branch
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        sequence_axis: Literal["height", "width"] = "height",
        use_diacritic_branch: bool = False,
    ):
        super().__init__()

        self.use_diacritic_branch = use_diacritic_branch

        self.patch_embed = SequencePatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            sequence_axis=sequence_axis,
        )

        self.backbone = SVTRBackbone(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        if self.use_diacritic_branch:
            self.detail_branch = DiacriticAwareBranch(
                in_channels=in_channels,
                embed_dim=embed_dim,
                sequence_axis=sequence_axis,
                dropout=dropout,
            )

            self.fusion = CrossAttentionFusion(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
            )
        else:
            self.detail_branch = None
            self.fusion = None

        self.head = CTCHead(
            in_channels=embed_dim,
            num_classes=num_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]

        Returns:
            logits: [B, T, num_classes]
        """
        global_seq = self.patch_embed(images)
        global_feat = self.backbone(global_seq)

        if self.use_diacritic_branch:
            detail_feat = self.detail_branch(images)
            global_feat = self.fusion(global_feat, detail_feat)

        logits = self.head(global_feat)
        return logits
