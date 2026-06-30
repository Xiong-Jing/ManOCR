from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from manchu_ocr.models.registry import RECOGNIZERS
from manchu_ocr.models.recognition.branches.diacritic_aware_branch import DiacriticAwareBranch
from manchu_ocr.models.recognition.heads.ctc_head import CTCHead
from manchu_ocr.models.recognition.modules.cross_attention_fusion import CrossAttentionFusion


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SVTRPatchEmbed(nn.Module):
    """
    SVTR patch embedding.

    Input:
        [B, C, H, W]

    Output:
        [B, embed_dim, H / patch_h, W / patch_w]
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        patch_size: tuple[int, int] = (4, 8),
        embed_type: Literal["patch", "conv"] = "conv",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_type = embed_type

        if embed_type == "patch":
            self.proj = ConvBNAct(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            )
        elif embed_type == "conv":
            # Gradual downsampling preserves small Manchu details better than a
            # single large-stride patch projection. For patch_size=[2, 16],
            # this produces the same output scale: H/2, W/16.
            if tuple(patch_size) != (2, 16):
                raise ValueError(
                    "conv patch embedding currently expects patch_size=(2, 16), "
                    f"got {patch_size}"
                )

            c1 = max(embed_dim // 4, 32)
            c2 = max(embed_dim // 2, 64)

            self.proj = nn.Sequential(
                ConvBNAct(in_channels, c1, kernel_size=3, stride=(1, 2), padding=1),
                ConvBNAct(c1, c2, kernel_size=3, stride=(1, 2), padding=1),
                ConvBNAct(c2, embed_dim, kernel_size=3, stride=(2, 2), padding=1),
                ConvBNAct(embed_dim, embed_dim, kernel_size=3, stride=(1, 2), padding=1),
            )
        else:
            raise ValueError(f"Unsupported patch embed_type: {embed_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Learned2DPositionEmbedding(nn.Module):
    """
    Learnable 2D positional embedding for the SVTR feature map.

    CTC recognition is sensitive to vertical order. Without positional
    information, global attention can over-mix spatial tokens and weaken the
    monotonic alignment required by CTC.
    """

    def __init__(
        self,
        channels: int,
        max_height: int,
        max_width: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, channels, max_height, max_width))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pos_embed

        if pos.shape[-2:] != x.shape[-2:]:
            pos = F.interpolate(
                pos,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return self.dropout(x + pos)


class SpatialSequencePooling(nn.Module):
    """
    Convert a 2D feature map into a CTC sequence.

    Mean pooling is fast, but it can suppress small Manchu details such as dots
    and circles. Attention pooling lets the recognizer learn which positions on
    the collapsed axis should contribute more to each sequence step.
    """

    def __init__(
        self,
        channels: int,
        sequence_axis: Literal["height", "width"] = "height",
        pooling_type: Literal["mean", "attention", "gated_attention"] = "mean",
    ):
        super().__init__()

        if sequence_axis not in {"height", "width"}:
            raise ValueError(f"Unsupported sequence_axis: {sequence_axis}")

        if pooling_type not in {"mean", "attention", "gated_attention"}:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        self.sequence_axis = sequence_axis
        self.pooling_type = pooling_type

        if pooling_type in {"attention", "gated_attention"}:
            hidden_channels = max(channels // 4, 32)
            self.score = nn.Sequential(
                nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )
        else:
            self.score = None

        if pooling_type == "gated_attention":
            # Start close to mean pooling and let training increase attention
            # selectivity only when it helps.
            self.attention_gate = nn.Parameter(torch.tensor(-1.0))
        else:
            self.attention_gate = None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            if self.sequence_axis == "height":
                seq = feat.mean(dim=3)
            else:
                seq = feat.mean(dim=2)
        else:
            if self.sequence_axis == "height":
                mean_seq = feat.mean(dim=3)
            else:
                mean_seq = feat.mean(dim=2)

            score = self.score(feat)

            if self.sequence_axis == "height":
                weights = torch.softmax(score, dim=3)
                attn_seq = (feat * weights).sum(dim=3)
            else:
                weights = torch.softmax(score, dim=2)
                attn_seq = (feat * weights).sum(dim=2)

            if self.pooling_type == "gated_attention":
                gate = torch.sigmoid(self.attention_gate)
                seq = mean_seq + gate * (attn_seq - mean_seq)
            else:
                seq = attn_seq

        return seq.transpose(1, 2).contiguous()


class SequenceContextEncoder(nn.Module):
    """
    Add 1D sequence context before the CTC classifier.

    CTC decoding is sensitive to local character confusions. A lightweight
    sequence encoder helps the classifier use neighboring vertical context
    without changing the image backbone.
    """

    def __init__(
        self,
        embed_dim: int,
        context_type: Literal["none", "bilstm", "transformer"] = "none",
        hidden_dim: int | None = None,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        residual_scale: float = 0.5,
    ):
        super().__init__()

        if context_type not in {"none", "bilstm", "transformer"}:
            raise ValueError(f"Unsupported sequence context type: {context_type}")

        self.context_type = context_type

        if context_type == "none":
            self.encoder = nn.Identity()
            self.proj = nn.Identity()
            self.norm = nn.Identity()
            self.dropout = nn.Identity()
            self.residual_scale = None
        elif context_type == "bilstm":
            hidden_dim = int(hidden_dim or embed_dim)
            self.norm = nn.LayerNorm(embed_dim)
            self.encoder = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.proj = nn.Linear(hidden_dim * 2, embed_dim)
            self.dropout = nn.Dropout(dropout)
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=layer,
                num_layers=num_layers,
            )
            self.proj = nn.Identity()
            self.norm = nn.Identity()
            self.dropout = nn.Identity()
            self.residual_scale = None

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if self.context_type == "none":
            return seq

        if self.context_type == "bilstm":
            context, _ = self.encoder(self.norm(seq))
            context = self.proj(context)
            return seq + self.residual_scale * self.dropout(context)

        return self.encoder(seq)


class SequenceRefinementBlock(nn.Module):
    """
    Local 1D refinement for CTC features.

    BiLSTM/attention captures long-range context, while this block focuses on
    neighboring character-level confusions. The depthwise convolution preserves
    monotonic order and is cheap enough for the full ablation suite.
    """

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        dropout: float = 0.05,
        residual_scale: float = 0.5,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("sequence refinement kernel_size should be odd.")

        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.local_mixer = nn.Sequential(
            nn.Conv1d(
                embed_dim,
                embed_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=embed_dim,
                bias=False,
            ),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.local_residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.ffn_residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        local_in = self.norm1(seq).transpose(1, 2).contiguous()
        local_out = self.local_mixer(local_in).transpose(1, 2).contiguous()
        seq = seq + self.local_residual_scale * local_out
        seq = seq + self.ffn_residual_scale * self.ffn(self.norm2(seq))
        return seq


class SequenceRefiner(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 0,
        kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        dropout: float = 0.05,
        residual_scale: float = 0.5,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                SequenceRefinementBlock(
                    embed_dim=embed_dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    residual_scale=residual_scale,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.blocks(seq)


class LocalMixer(nn.Module):
    """
    Local spatial mixing with depthwise convolution.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GlobalMixer(nn.Module):
    """
    Global token mixing with multi-head self-attention.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_axis: Literal["spatial", "height", "width"] = "spatial",
    ):
        super().__init__()

        if attention_axis not in {"spatial", "height", "width"}:
            raise ValueError(f"Unsupported attention_axis: {attention_axis}")

        self.attention_axis = attention_axis
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        if self.attention_axis == "spatial":
            seq = x.flatten(2).transpose(1, 2).contiguous()
            seq = self.norm(seq)

            out, _ = self.attn(seq, seq, seq, need_weights=False)

            out = out.transpose(1, 2).contiguous().view(b, c, h, w)
            return out

        if self.attention_axis == "height":
            # Vertical Manchu recognition mainly needs long-range context along
            # the reading direction. This reduces attention length from H*W to
            # H and is substantially faster than full 2D attention.
            seq = x.permute(0, 3, 2, 1).contiguous().view(b * w, h, c)
            target_shape = (b, w, h, c)
            inverse = "height"
        else:
            seq = x.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
            target_shape = (b, h, w, c)
            inverse = "width"

        seq = self.norm(seq)

        out, _ = self.attn(seq, seq, seq, need_weights=False)

        if inverse == "height":
            out = out.view(*target_shape).permute(0, 3, 2, 1).contiguous()
        else:
            out = out.view(*target_shape).permute(0, 3, 1, 2).contiguous()

        return out


class FeedForward2D(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_channels = int(channels * mlp_ratio)

        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SVTRBlock(nn.Module):
    """
    SVTR-style block:
        local/global mixer + feed-forward network.
    """

    def __init__(
        self,
        channels: int,
        mixer_type: Literal["local", "global"] = "local",
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        global_mixer_axis: Literal["spatial", "height", "width"] = "spatial",
    ):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(channels)

        if mixer_type == "local":
            self.mixer = LocalMixer(channels=channels)
        elif mixer_type == "global":
            self.mixer = GlobalMixer(
                channels=channels,
                num_heads=num_heads,
                dropout=dropout,
                attention_axis=global_mixer_axis,
            )
        else:
            raise ValueError(f"Unsupported mixer_type: {mixer_type}")

        self.norm2 = nn.BatchNorm2d(channels)

        self.ffn = FeedForward2D(
            channels=channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SVTR2DEncoder(nn.Module):
    """
    Multi-stage SVTR encoder.

    Current compact version:
        - keeps spatial resolution stable after patch embedding
        - supports local and global mixers
        - later can be extended with patch merging
    """

    def __init__(
        self,
        embed_dim: int = 128,
        depths: List[int] | tuple[int, ...] = (2, 2, 2),
        mixer_types: List[str] | tuple[str, ...] = ("local", "local", "global"),
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        global_mixer_axis: Literal["spatial", "height", "width"] = "spatial",
    ):
        super().__init__()

        if len(depths) != len(mixer_types):
            raise ValueError(
                f"depths and mixer_types should have same length, "
                f"got {len(depths)} and {len(mixer_types)}"
            )

        blocks = []

        for depth, mixer_type in zip(depths, mixer_types):
            for _ in range(depth):
                blocks.append(
                    SVTRBlock(
                        channels=embed_dim,
                        mixer_type=mixer_type,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        global_mixer_axis=global_mixer_axis,
                    )
                )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


@RECOGNIZERS.register("SVTROfficialRecognizer")
class SVTROfficialRecognizer(nn.Module):
    """
    SVTR recognizer for vertical Manchu word recognition.

    Input:
        [B, 3, H, W]

    Output:
        logits [B, T, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        embed_dim: int = 128,
        patch_size: tuple[int, int] = (4, 8),
        patch_embed_type: Literal["patch", "conv"] = "conv",
        depths: List[int] | tuple[int, ...] = (2, 2, 2),
        mixer_types: List[str] | tuple[str, ...] = ("local", "local", "global"),
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        global_mixer_axis: Literal["spatial", "height", "width"] = "spatial",
        sequence_axis: Literal["height", "width"] = "height",
        use_2d_pos_embed: bool = True,
        max_image_height: int = 256,
        max_image_width: int = 64,
        pos_dropout: float = 0.0,
        pooling_type: Literal["mean", "attention", "gated_attention"] = "mean",
        sequence_context_type: Literal["none", "bilstm", "transformer"] = "none",
        sequence_context_hidden_dim: int | None = None,
        sequence_context_layers: int = 1,
        sequence_context_dropout: float = 0.0,
        sequence_context_residual_scale: float = 0.5,
        sequence_refine_layers: int = 0,
        sequence_refine_kernel_size: int = 5,
        sequence_refine_mlp_ratio: float = 2.0,
        sequence_refine_dropout: float = 0.05,
        sequence_refine_residual_scale: float = 0.5,
        use_diacritic_branch: bool = False,
        dab_dropout: float | None = None,
        dab_sobel_trainable: bool = True,
        dab_edge_normalize: bool = False,
        dab_edge_scale: float = 1.0,
        dab_fusion_attn_scale: float = 0.1,
        dab_fusion_ffn_scale: float = 0.1,
        dab_fusion_max_scale: float = 0.15,
        dab_fusion_gate_bias: float = -2.0,
    ):
        super().__init__()

        if sequence_axis not in {"height", "width"}:
            raise ValueError(f"Unsupported sequence_axis: {sequence_axis}")

        self.sequence_axis = sequence_axis
        self.use_diacritic_branch = use_diacritic_branch
        patch_size = tuple(patch_size)

        self.patch_embed = SVTRPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            embed_type=patch_embed_type,
        )

        if use_2d_pos_embed:
            max_feat_height = max(1, int(max_image_height) // int(patch_size[0]))
            max_feat_width = max(1, int(max_image_width) // int(patch_size[1]))
            self.pos_embed = Learned2DPositionEmbedding(
                channels=embed_dim,
                max_height=max_feat_height,
                max_width=max_feat_width,
                dropout=pos_dropout,
            )
        else:
            self.pos_embed = nn.Identity()

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

        if self.use_diacritic_branch:
            detail_dropout = dropout if dab_dropout is None else float(dab_dropout)

            self.detail_branch = DiacriticAwareBranch(
                in_channels=in_channels,
                embed_dim=embed_dim,
                sequence_axis=sequence_axis,
                dropout=detail_dropout,
                sobel_trainable=dab_sobel_trainable,
                normalize_edges=dab_edge_normalize,
                edge_scale=dab_edge_scale,
                sequence_downsample=patch_size[0] if sequence_axis == "height" else patch_size[1],
            )

            self.fusion = CrossAttentionFusion(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=detail_dropout,
                mlp_ratio=mlp_ratio,
                attn_residual_scale=dab_fusion_attn_scale,
                ffn_residual_scale=dab_fusion_ffn_scale,
                max_residual_scale=dab_fusion_max_scale,
                gate_bias=dab_fusion_gate_bias,
            )
        else:
            self.detail_branch = None
            self.fusion = None

        self.sequence_context = SequenceContextEncoder(
            embed_dim=embed_dim,
            context_type=sequence_context_type,
            hidden_dim=sequence_context_hidden_dim,
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
            mlp_ratio=sequence_refine_mlp_ratio,
            dropout=sequence_refine_dropout,
            residual_scale=sequence_refine_residual_scale,
        )

        self.head = CTCHead(
            in_channels=embed_dim,
            num_classes=num_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(images)
        feat = self.pos_embed(feat)
        feat = self.encoder(feat)

        seq = self.sequence_pool(feat)

        if self.use_diacritic_branch:
            detail_seq = self.detail_branch(images)
            seq = self.fusion(seq, detail_seq)

        seq = self.sequence_context(seq)
        seq = self.sequence_refiner(seq)

        logits = self.head(seq)
        return logits
