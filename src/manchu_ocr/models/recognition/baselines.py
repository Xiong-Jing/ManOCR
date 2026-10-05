from typing import Literal

import torch
import torch.nn as nn

from manchu_ocr.models.recognition.heads.ctc_head import CTCHead
from manchu_ocr.models.registry import RECOGNIZERS


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SequenceProjector(nn.Module):
    def __init__(
        self,
        sequence_axis: Literal["height", "width"] = "height",
        pooling_type: Literal["mean", "attention"] = "mean",
        channels: int = 256,
    ):
        super().__init__()

        if sequence_axis not in {"height", "width"}:
            raise ValueError(f"Unsupported sequence_axis: {sequence_axis}")

        if pooling_type not in {"mean", "attention"}:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        self.sequence_axis = sequence_axis
        self.pooling_type = pooling_type

        if pooling_type == "attention":
            hidden_channels = max(channels // 4, 32)
            self.score = nn.Sequential(
                nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )
        else:
            self.score = None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            if self.sequence_axis == "height":
                seq = feat.mean(dim=3)
            else:
                seq = feat.mean(dim=2)
        else:
            score = self.score(feat)

            if self.sequence_axis == "height":
                weight = torch.softmax(score, dim=3)
                seq = (feat * weight).sum(dim=3)
            else:
                weight = torch.softmax(score, dim=2)
                seq = (feat * weight).sum(dim=2)

        return seq.transpose(1, 2).contiguous()


class LearnedSequencePosition(nn.Module):
    def __init__(self, channels: int, max_length: int = 512, dropout: float = 0.0):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, channels))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= self.pos_embed.shape[1]:
            pos = self.pos_embed[:, : x.shape[1], :]
        else:
            pos = self.pos_embed.transpose(1, 2)
            pos = torch.nn.functional.interpolate(
                pos,
                size=x.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        return self.dropout(x + pos)


@RECOGNIZERS.register("CRNNRecognizer")
class CRNNRecognizer(nn.Module):
    """
    CRNN baseline: CNN feature extractor + BiLSTM + CTC classifier.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        cnn_channels: int = 256,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        sequence_axis: Literal["height", "width"] = "height",
    ):
        super().__init__()

        c1 = max(cnn_channels // 4, 64)
        c2 = max(cnn_channels // 2, 128)

        self.cnn = nn.Sequential(
            ConvBNAct(in_channels, c1, kernel_size=3, stride=1, padding=1),
            ConvBNAct(c1, c1, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(c1, c2, kernel_size=3, stride=1, padding=1),
            ConvBNAct(c2, c2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(c2, cnn_channels, kernel_size=3, stride=1, padding=1),
            ConvBNAct(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
        )

        self.project = SequenceProjector(
            sequence_axis=sequence_axis,
            pooling_type="mean",
            channels=cnn_channels,
        )

        self.rnn = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(rnn_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden * 2, cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = CTCHead(cnn_channels, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(images)
        seq = self.project(feat)
        seq, _ = self.rnn(seq)
        seq = self.proj(seq)
        return self.head(seq)


@RECOGNIZERS.register("PARSeqCTCRecognizer")
class PARSeqCTCRecognizer(nn.Module):
    """
    PARSeq-style transformer baseline adapted to CTC output.

    It uses patch embedding, learned sequence positions, and stacked
    Transformer encoder blocks. The output is CTC-compatible for the common
    project evaluation protocol.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        embed_dim: int = 256,
        patch_size: tuple[int, int] = (2, 16),
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        sequence_axis: Literal["height", "width"] = "height",
        pooling_type: Literal["mean", "attention"] = "attention",
    ):
        super().__init__()

        self.patch_embed = nn.Sequential(
            ConvBNAct(in_channels, embed_dim // 2, kernel_size=3, stride=(1, 2), padding=1),
            ConvBNAct(embed_dim // 2, embed_dim, kernel_size=3, stride=(2, 2), padding=1),
            ConvBNAct(embed_dim, embed_dim, kernel_size=3, stride=(1, 4), padding=1),
        )

        expected_patch = tuple(patch_size)
        if expected_patch != (2, 16):
            raise ValueError(f"PARSeqCTCRecognizer expects patch_size=(2, 16), got {patch_size}")

        self.project = SequenceProjector(
            sequence_axis=sequence_axis,
            pooling_type=pooling_type,
            channels=embed_dim,
        )
        self.pos = LearnedSequencePosition(embed_dim, max_length=max_seq_len, dropout=dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = CTCHead(embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(images)
        seq = self.project(feat)
        seq = self.pos(seq)
        seq = self.encoder(seq)
        seq = self.norm(seq)
        return self.head(seq)


@RECOGNIZERS.register("ABINetCTCRecognizer")
class ABINetCTCRecognizer(nn.Module):
    """
    ABINet-style baseline adapted to CTC.

    It combines a visual transformer branch with a lightweight autonomous
    language refinement branch, then fuses both streams before CTC prediction.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        embed_dim: int = 256,
        patch_size: tuple[int, int] = (2, 16),
        visual_depth: int = 6,
        language_depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        sequence_axis: Literal["height", "width"] = "height",
    ):
        super().__init__()

        if tuple(patch_size) != (2, 16):
            raise ValueError(f"ABINetCTCRecognizer expects patch_size=(2, 16), got {patch_size}")

        self.patch_embed = nn.Sequential(
            ConvBNAct(in_channels, embed_dim // 2, kernel_size=3, stride=(1, 2), padding=1),
            ConvBNAct(embed_dim // 2, embed_dim, kernel_size=3, stride=(2, 2), padding=1),
            ConvBNAct(embed_dim, embed_dim, kernel_size=3, stride=(1, 4), padding=1),
        )

        self.project = SequenceProjector(
            sequence_axis=sequence_axis,
            pooling_type="attention",
            channels=embed_dim,
        )
        self.pos = LearnedSequencePosition(embed_dim, max_length=max_seq_len, dropout=dropout)

        visual_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(visual_layer, num_layers=visual_depth)

        language_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.language_encoder = nn.TransformerEncoder(language_layer, num_layers=language_depth)

        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = CTCHead(embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(images)
        seq = self.project(feat)
        seq = self.pos(seq)

        visual_seq = self.visual_encoder(seq)
        language_seq = self.language_encoder(visual_seq.detach()) + visual_seq

        gate = self.fusion_gate(torch.cat([visual_seq, language_seq], dim=-1))
        fused = visual_seq + gate * (language_seq - visual_seq)
        fused = self.dropout(self.norm(fused))
        return self.head(fused)
