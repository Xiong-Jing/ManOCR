from pathlib import Path

import torch

from manchu_ocr.losses.ctc_loss import CTCLossWrapper
from manchu_ocr.losses.nrtr_loss import NRTRLoss
from manchu_ocr.models.recognition.advanced_baselines import SVTRv2NRTRRecognizer
from manchu_ocr.models.recognition.heads.nrtr_head import NRTRDecoderHead
from manchu_ocr.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_small_model() -> SVTRv2NRTRRecognizer:
    return SVTRv2NRTRRecognizer(
        num_classes=6,
        in_channels=3,
        embed_dim=32,
        patch_size=(4, 4),
        patch_embed_type="patch",
        depths=(1,),
        mixer_types=("global",),
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        global_mixer_axis="height",
        sequence_axis="height",
        max_image_height=16,
        max_image_width=16,
        pooling_type="gated_attention",
        multi_size_scales=(1.0,),
        rearrange_layers=1,
        semantic_tokens=4,
        semantic_residual_scale=0.2,
        sequence_context_type="none",
        sequence_context_hidden_dim=32,
        sequence_context_layers=1,
        sequence_context_dropout=0.0,
        sequence_context_residual_scale=0.1,
        sequence_refine_layers=1,
        sequence_refine_kernel_size=3,
        sequence_refine_dropout=0.0,
        nrtr_decoder_dim=32,
        nrtr_num_layers=1,
        nrtr_num_heads=4,
        nrtr_ffn_dim=64,
        nrtr_dropout=0.0,
        nrtr_max_text_length=8,
    )


def test_nrtr_teacher_forcing_sequences_preserve_ctc_character_indices():
    head = NRTRDecoderHead(
        memory_dim=16,
        num_ctc_classes=6,
        decoder_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        max_text_length=8,
    )
    inputs, targets, padding_mask = head.build_teacher_forcing_sequences(
        targets=torch.tensor([1, 2, 3]),
        target_lengths=torch.tensor([2, 1]),
    )

    assert inputs.tolist() == [[head.bos_idx, 1, 2], [head.bos_idx, 3, 0]]
    assert targets.tolist() == [[1, 2, head.eos_idx], [3, head.eos_idx, 0]]
    assert padding_mask.tolist() == [[False, False, False], [False, False, True]]


def test_svtrv2_nrtr_joint_training_and_ctc_only_inference():
    model = build_small_model()
    images = torch.randn(2, 3, 16, 16)
    targets = torch.tensor([1, 2, 3], dtype=torch.long)
    target_lengths = torch.tensor([2, 1], dtype=torch.long)

    train_outputs = model.forward_train(images, targets, target_lengths)
    ctc_loss = CTCLossWrapper()(
        train_outputs["ctc_logits"],
        targets,
        target_lengths,
    )
    nrtr_loss = NRTRLoss(pad_idx=model.nrtr_pad_idx)(
        train_outputs["nrtr_logits"],
        train_outputs["nrtr_targets"],
    )
    joint_loss = ctc_loss + nrtr_loss
    joint_loss.backward()

    assert torch.isfinite(joint_loss)
    assert train_outputs["ctc_logits"].shape == (2, 4, 6)
    assert train_outputs["nrtr_logits"].shape == (2, 3, 8)
    assert any(parameter.grad is not None for parameter in model.nrtr_head.parameters())

    model.eval()
    with torch.no_grad():
        inference_output = model(images)
    assert isinstance(inference_output, torch.Tensor)
    assert inference_output.shape == train_outputs["ctc_logits"].shape


def test_svtrv2_nrtr_validation_and_test_protocol_matches_svtrv2():
    svtrv2_cfg = load_yaml(
        PROJECT_ROOT / "configs" / "recognition" / "svtrv2_baseline.yaml"
    )
    nrtr_cfg = load_yaml(
        PROJECT_ROOT / "configs" / "recognition" / "svtrv2_nrtr_baseline.yaml"
    )

    assert nrtr_cfg["runtime"] == svtrv2_cfg["runtime"]
    assert nrtr_cfg["data"] == svtrv2_cfg["data"]
    assert nrtr_cfg["decode"] == svtrv2_cfg["decode"]
    assert nrtr_cfg.get("metrics", {}) == svtrv2_cfg.get("metrics", {})
    assert nrtr_cfg["experiment"]["paths_config"] == svtrv2_cfg["experiment"][
        "paths_config"
    ]

    validation_keys = (
        "val_interval",
        "rerank_val_interval",
        "validate_during_training",
        "best_metric",
    )
    for key in validation_keys:
        assert nrtr_cfg["train"][key] == svtrv2_cfg["train"][key]

    nrtr_only_model_keys = {key for key in nrtr_cfg["model"] if key.startswith("nrtr_")}
    shared_nrtr_model = {
        key: value
        for key, value in nrtr_cfg["model"].items()
        if key != "name" and key not in nrtr_only_model_keys
    }
    shared_svtrv2_model = {
        key: value for key, value in svtrv2_cfg["model"].items() if key != "name"
    }
    assert shared_nrtr_model == shared_svtrv2_model

    for key, value in svtrv2_cfg["loss"].items():
        assert nrtr_cfg["loss"][key] == value
