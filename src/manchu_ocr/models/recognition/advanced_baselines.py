from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from manchu_ocr.models.recognition.heads.ctc_head import CTCHead
from manchu_ocr.models.recognition.heads.nrtr_head import NRTRDecoderHead
from manchu_ocr.models.recognition.svtr_official import (
    Learned2DPositionEmbedding,
    SequenceContextEncoder,
    SequenceRefiner,
    SpatialSequencePooling,
    SVTR2DEncoder,
    SVTRPatchEmbed,
)
from manchu_ocr.models.registry import RECOGNIZERS


class FeatureRearrangementBlock(nn.Module):
    """
    SVTRv2-style feature rearrangement.

    The block mixes vertical and horizontal local structures before CTC
    sequence projection. This is useful for irregular glyph width and long
    vertical text, while remaining weaker than the project's dedicated DAB and
    Lortho modules.
    """

    def __init__(
        self,
        channels: int,
        vertical_kernel: int = 5,
        horizontal_kernel: int = 3,
        dropout: float = 0.0,
        residual_scale: float = 0.25,
    ):
        super().__init__()

        if vertical_kernel % 2 == 0:
            raise ValueError("vertical_kernel should be odd.")
        if horizontal_kernel % 2 == 0:
            raise ValueError("horizontal_kernel should be odd.")

        self.norm = nn.BatchNorm2d(channels)
        self.vertical = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(vertical_kernel, 1),
                padding=(vertical_kernel // 2, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.horizontal = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, horizontal_kernel),
                padding=(0, horizontal_kernel // 2),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.norm(feat)
        rearranged = self.fuse(torch.cat([self.vertical(x), self.horizontal(x)], dim=1))
        return feat + self.residual_scale * rearranged


class SemanticGuidanceBlock(nn.Module):
    """
    Semantic guidance module for SVTRv2-style comparison.

    It uses learnable semantic tokens to collect sequence context and then
    cross-attends CTC features back to those tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_semantic_tokens: int = 16,
        dropout: float = 0.0,
        residual_scale: float = 0.25,
    ):
        super().__init__()

        self.semantic_tokens = nn.Parameter(torch.zeros(1, num_semantic_tokens, embed_dim))
        nn.init.trunc_normal_(self.semantic_tokens, std=0.02)

        self.collect_norm = nn.LayerNorm(embed_dim)
        self.semantic_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.guided_norm = nn.LayerNorm(embed_dim)
        self.guided_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size = seq.shape[0]
        tokens = self.semantic_tokens.expand(batch_size, -1, -1)

        semantic, _ = self.semantic_attn(
            tokens,
            self.collect_norm(seq),
            self.collect_norm(seq),
            need_weights=False,
        )

        guided, _ = self.guided_attn(
            self.guided_norm(seq),
            semantic,
            semantic,
            need_weights=False,
        )

        gate = self.gate(torch.cat([seq, guided], dim=-1))
        seq = seq + self.residual_scale * self.dropout(gate * guided)
        return self.out_norm(seq)


@RECOGNIZERS.register("SVTRv2Recognizer")
class SVTRv2Recognizer(nn.Module):
    """
    Project-native SVTRv2 comparison recognizer.

    It includes the main SVTRv2 ideas used in the paper comparison discussion:
    multi-size feature fusion, feature rearrangement, semantic guidance, and a
    CTC-compatible output head.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        embed_dim: int = 384,
        patch_size: tuple[int, int] = (2, 16),
        patch_embed_type: Literal["patch", "conv"] = "conv",
        depths: list[int] | tuple[int, ...] = (3, 4, 3),
        mixer_types: list[str] | tuple[str, ...] = ("local", "local", "global"),
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.04,
        global_mixer_axis: Literal["spatial", "height", "width"] = "height",
        sequence_axis: Literal["height", "width"] = "height",
        max_image_height: int = 256,
        max_image_width: int = 128,
        pooling_type: Literal["mean", "attention", "gated_attention"] = "gated_attention",
        multi_size_scales: list[float] | tuple[float, ...] = (1.0, 0.875),
        rearrange_layers: int = 2,
        semantic_tokens: int = 16,
        semantic_residual_scale: float = 0.22,
        sequence_context_type: Literal["none", "bilstm", "transformer"] = "bilstm",
        sequence_context_hidden_dim: int | None = None,
        sequence_context_layers: int = 1,
        sequence_context_dropout: float = 0.04,
        sequence_context_residual_scale: float = 0.16,
        sequence_refine_layers: int = 1,
        sequence_refine_kernel_size: int = 5,
        sequence_refine_dropout: float = 0.04,
    ):
        super().__init__()

        patch_size = tuple(patch_size)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.multi_size_scales = tuple(float(scale) for scale in multi_size_scales)

        self.patch_embed = SVTRPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            embed_type=patch_embed_type,
        )

        self.pos_embed = Learned2DPositionEmbedding(
            channels=embed_dim,
            max_height=max(1, int(max_image_height) // int(patch_size[0])),
            max_width=max(1, int(max_image_width) // int(patch_size[1])),
            dropout=0.0,
        )

        self.encoder = SVTR2DEncoder(
            embed_dim=embed_dim,
            depths=depths,
            mixer_types=mixer_types,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            global_mixer_axis=global_mixer_axis,
        )

        self.rearrange = nn.Sequential(
            *[
                FeatureRearrangementBlock(
                    channels=embed_dim,
                    dropout=dropout,
                    residual_scale=0.20,
                )
                for _ in range(rearrange_layers)
            ]
        )

        self.sequence_pool = SpatialSequencePooling(
            channels=embed_dim,
            sequence_axis=sequence_axis,
            pooling_type=pooling_type,
        )
        self.semantic_guidance = SemanticGuidanceBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_semantic_tokens=semantic_tokens,
            dropout=dropout,
            residual_scale=semantic_residual_scale,
        )
        self.sequence_context = SequenceContextEncoder(
            embed_dim=embed_dim,
            context_type=sequence_context_type,
            hidden_dim=sequence_context_hidden_dim or embed_dim,
            num_layers=sequence_context_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=sequence_context_dropout,
            residual_scale=sequence_context_residual_scale,
        )
        self.sequence_refiner = SequenceRefiner(
            embed_dim=embed_dim,
            num_layers=sequence_refine_layers,
            kernel_size=sequence_refine_kernel_size,
            dropout=sequence_refine_dropout,
            residual_scale=0.10,
        )
        self.head = CTCHead(embed_dim, num_classes)

    def _extract_multi_size_features(self, images: torch.Tensor) -> torch.Tensor:
        base_feat = self.patch_embed(images)
        target_size = base_feat.shape[-2:]
        feats = [base_feat]

        height, width = images.shape[-2:]
        for scale in self.multi_size_scales:
            if abs(scale - 1.0) < 1e-6:
                continue

            scaled_width = max(16, int(round(width * scale)))
            scaled = F.interpolate(
                images,
                size=(height, scaled_width),
                mode="bilinear",
                align_corners=False,
            )
            feat = self.patch_embed(scaled)
            feat = F.interpolate(
                feat,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            feats.append(feat)

        return torch.stack(feats, dim=0).mean(dim=0)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        feat = self._extract_multi_size_features(images)
        feat = self.pos_embed(feat)
        feat = self.encoder(feat)
        feat = self.rearrange(feat)

        seq = self.sequence_pool(feat)
        seq = self.semantic_guidance(seq)
        seq = self.sequence_context(seq)
        seq = self.sequence_refiner(seq)
        return seq

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        seq = self.forward_features(images)
        return self.head(seq)


@RECOGNIZERS.register("SVTRv2NRTRRecognizer")
class SVTRv2NRTRRecognizer(SVTRv2Recognizer):
    """SVTRv2 with an auxiliary autoregressive NRTR decoder.

    SVTRv2 visual features are shared by a CTC head and an NRTR Transformer
    decoder during training. ``forward`` intentionally returns only CTC logits,
    keeping validation, held-out testing, and deployment identical to the
    original SVTRv2 baseline. ``forward_train`` exposes both branches for the
    joint CTC + NRTR objective.
    """

    def __init__(
        self,
        num_classes: int,
        nrtr_decoder_dim: int = 384,
        nrtr_num_layers: int = 2,
        nrtr_num_heads: int = 8,
        nrtr_ffn_dim: int = 1536,
        nrtr_dropout: float = 0.1,
        nrtr_max_text_length: int = 128,
        **svtrv2_kwargs,
    ) -> None:
        super().__init__(num_classes=num_classes, **svtrv2_kwargs)
        self.nrtr_head = NRTRDecoderHead(
            memory_dim=self.embed_dim,
            num_ctc_classes=self.num_classes,
            decoder_dim=int(nrtr_decoder_dim),
            num_layers=int(nrtr_num_layers),
            num_heads=int(nrtr_num_heads),
            ffn_dim=int(nrtr_ffn_dim),
            dropout=float(nrtr_dropout),
            max_text_length=int(nrtr_max_text_length),
        )

    @property
    def nrtr_pad_idx(self) -> int:
        return self.nrtr_head.pad_idx

    def forward_train(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sequence_features = self.forward_features(images)
        ctc_logits = self.head(sequence_features)
        decoder_inputs, decoder_targets, padding_mask = (
            self.nrtr_head.build_teacher_forcing_sequences(
                targets=targets,
                target_lengths=target_lengths,
            )
        )
        nrtr_logits = self.nrtr_head(
            memory=sequence_features,
            decoder_inputs=decoder_inputs,
            padding_mask=padding_mask,
        )
        return {
            "ctc_logits": ctc_logits,
            "nrtr_logits": nrtr_logits,
            "nrtr_targets": decoder_targets,
        }


class CharacterAwareConstraintEncoder(nn.Module):
    """
    DCM-style character-aware constraint encoder.

    Learnable character prototypes help separate visually similar Manchu
    characters before CTC classification.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.04,
        residual_scale: float = 0.25,
    ):
        super().__init__()

        num_characters = max(int(num_classes) - 1, 1)
        self.char_prototypes = nn.Parameter(torch.zeros(1, num_characters, embed_dim))
        nn.init.trunc_normal_(self.char_prototypes, std=0.02)

        self.norm = nn.LayerNorm(embed_dim)
        self.prototype_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size = seq.shape[0]
        prototypes = self.char_prototypes.expand(batch_size, -1, -1)
        query = self.norm(seq)

        char_context, _ = self.prototype_attn(
            query,
            prototypes,
            prototypes,
            need_weights=False,
        )

        fused = self.fuse(torch.cat([seq, char_context], dim=-1))
        return self.out_norm(seq + self.residual_scale * fused)


@RECOGNIZERS.register("DCMRecognizer")
class DCMRecognizer(nn.Module):
    """
    Project-native DCM comparison recognizer.

    The model focuses on character discriminability through a
    Character-Aware Constraint Encoder before CTC prediction.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        embed_dim: int = 384,
        patch_size: tuple[int, int] = (2, 16),
        patch_embed_type: Literal["patch", "conv"] = "conv",
        depths: list[int] | tuple[int, ...] = (3, 3, 3),
        mixer_types: list[str] | tuple[str, ...] = ("local", "local", "global"),
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.04,
        global_mixer_axis: Literal["spatial", "height", "width"] = "height",
        sequence_axis: Literal["height", "width"] = "height",
        max_image_height: int = 256,
        max_image_width: int = 128,
        pooling_type: Literal["mean", "attention", "gated_attention"] = "attention",
        constraint_layers: int = 2,
        constraint_residual_scale: float = 0.22,
        sequence_context_type: Literal["none", "bilstm", "transformer"] = "bilstm",
        sequence_context_hidden_dim: int | None = None,
        sequence_context_dropout: float = 0.04,
    ):
        super().__init__()

        patch_size = tuple(patch_size)
        self.patch_embed = SVTRPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            embed_type=patch_embed_type,
        )
        self.pos_embed = Learned2DPositionEmbedding(
            channels=embed_dim,
            max_height=max(1, int(max_image_height) // int(patch_size[0])),
            max_width=max(1, int(max_image_width) // int(patch_size[1])),
            dropout=0.0,
        )
        self.encoder = SVTR2DEncoder(
            embed_dim=embed_dim,
            depths=depths,
            mixer_types=mixer_types,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            global_mixer_axis=global_mixer_axis,
        )
        self.sequence_pool = SpatialSequencePooling(
            channels=embed_dim,
            sequence_axis=sequence_axis,
            pooling_type=pooling_type,
        )
        self.sequence_context = SequenceContextEncoder(
            embed_dim=embed_dim,
            context_type=sequence_context_type,
            hidden_dim=sequence_context_hidden_dim or embed_dim,
            num_layers=1,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=sequence_context_dropout,
            residual_scale=0.14,
        )
        self.constraint = nn.Sequential(
            *[
                CharacterAwareConstraintEncoder(
                    embed_dim=embed_dim,
                    num_classes=num_classes,
                    num_heads=num_heads,
                    dropout=dropout,
                    residual_scale=constraint_residual_scale,
                )
                for _ in range(constraint_layers)
            ]
        )
        self.head = CTCHead(embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(images)
        feat = self.pos_embed(feat)
        feat = self.encoder(feat)

        seq = self.sequence_pool(feat)
        seq = self.sequence_context(seq)
        seq = self.constraint(seq)
        return self.head(seq)
