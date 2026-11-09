from pathlib import Path
import yaml


PATHS_CONFIG = "configs/paths/remote_server.yaml"
REMOTE_REC_ROOT = "${OCR_MANCHU_REC_ROOT:-/root/code/Manchu_Recognition_Data}"


def write_yaml(path: str, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[OK] wrote {path}")


def recognition_config(
    exp_name: str,
    use_dab: bool = False,
    use_lortho: bool = False,
) -> dict:
    # 4090 formal recognition setting.
    #
    # The same backbone/training budget is kept across ablation configs so the
    # comparison remains defensible. DAB/Lortho configs only enable their own
    # modules and module-specific loss terms.
    loss = {
        "blank_idx": 0,
        "zero_infinity": True,
        "use_orthographic_loss": use_lortho,
        "use_alignment_loss": False,
        "lambda_align": 0.0,
        "lambda_align_final": 0.0,
        "lambda_align_decay_start_epoch": 80,
        "lambda_align_decay_end_epoch": 160,
        "align_label_smoothing": 0.03,
        "lambda_align_activate_epoch": 5,
        "lambda_align_warmup_epochs": 20,
    }

    if use_lortho:
        loss.update(
            {
                "lambda_ortho": 0.1,
                "normalize_char_probs": True,
                "confidence_weighting": True,
                "confidence_power": 1.0,
                "min_confidence": 0.0,
                "transition_matrix": f"{REMOTE_REC_ROOT}/processed/transition_matrix.npy",
            }
        )

    model = {
        "name": "SVTROfficialRecognizer",
        "in_channels": 3,
        "embed_dim": 384,
        "patch_size": [2, 16],
        "patch_embed_type": "conv",
        "depths": [3, 4, 3],
        "mixer_types": ["local", "local", "global"],
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.04,
        "global_mixer_axis": "height",
        "sequence_axis": "height",
        "use_2d_pos_embed": True,
        "max_image_height": 256,
        "max_image_width": 128,
        "pos_dropout": 0.0,
        "pooling_type": "gated_attention",
        "sequence_context_type": "bilstm",
        "sequence_context_hidden_dim": 384,
        "sequence_context_layers": 1,
        "sequence_context_dropout": 0.04,
        "sequence_context_residual_scale": 0.18,
        "sequence_refine_layers": 2,
        "sequence_refine_kernel_size": 5,
        "sequence_refine_mlp_ratio": 2.0,
        "sequence_refine_dropout": 0.04,
        "sequence_refine_residual_scale": 0.12,
        "use_diacritic_branch": use_dab,
    }

    if use_dab:
        model.update(
            {
                "dab_dropout": 0.01,
                "dab_sobel_trainable": True,
                "dab_edge_normalize": True,
                "dab_edge_scale": 0.5,
                "dab_fusion_attn_scale": 0.06,
                "dab_fusion_ffn_scale": 0.015,
                "dab_fusion_max_scale": 0.12,
                "dab_fusion_gate_bias": -1.5,
            }
        )

    train_batch_size = 128
    grad_accum_steps = 1

    return {
        "experiment": {
            "protocol_version": "paper_v2_20261109",
            "name": exp_name,
            "task": "recognition",
            "paths_config": PATHS_CONFIG,
        },
        "runtime": {
            "device": "cuda",
            "allow_cpu": False,
        },
        "data": {
            "image_height": 256,
            "image_width": 128,
            "sequence_axis": "height",
            "batch_size": train_batch_size,
            "num_workers": 8,
            "augmentation": {
                "enabled": True,
                "rotation_degrees": 0.5,
                "translate_ratio": 0.008,
                "scale_min": 0.985,
                "scale_max": 1.015,
                "brightness": 0.04,
                "contrast": 0.04,
            },
        },
        "model": model,
        "loss": loss,
        "decode": {
            "use_lexicon": False,
            "lexicon_source": "train",
            "max_edit_distance": 2,
            "length_delta": 2,
            "min_word_length": 2,
            "use_ctc_lexicon_rerank": False,
            "ctc_rerank_max_edit_distance": 2,
            "ctc_rerank_length_delta": 2,
            "ctc_rerank_min_word_length": 2,
            "ctc_rerank_max_candidates": 40,
            "ctc_rerank_prior_weight": 0.04,
        },
        "train": {
                "epochs": 200,
                "lr": 0.00006,
                "min_lr": 0.000001,
                "weight_decay": 0.000030,
                "amp": True,
                "seed": 42,
                "log_interval": 100,
                "val_interval": 1,
                "rerank_val_interval": 0,
                "validate_during_training": True,
                "grad_accum_steps": grad_accum_steps,
                "warmup_epochs": 10,
                "scheduler": "cosine",
                "grad_clip_norm": 1.0,
                "early_stop_patience": 0,
                "use_ema": True,
                "ema_decay": 0.999,
                "channels_last": True,
                "cudnn_benchmark": True,
                "allow_tf32": True,
                "best_metric": "word_accuracy",
            },
    }


def otp_lambda_sensitivity_config(exp_name: str, lambda_value: float) -> dict:
    """Build one controlled DAB + OTP-weight sensitivity configuration."""
    lambda_value = float(lambda_value)
    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative")

    use_otp = lambda_value > 0.0
    cfg = recognition_config(
        exp_name,
        use_dab=True,
        use_lortho=use_otp,
    )
    cfg["ablation"] = {
        "study": "otp_lambda_sensitivity",
        "parameter": "lambda_ortho",
        "value": lambda_value,
        "otp_enabled": use_otp,
        "fixed_component": "DAB",
        "validation_logic_source": "unified_strict",
    }

    if use_otp:
        cfg["loss"].update(
            {
                "lambda_ortho": lambda_value,
            }
        )
    return cfg


def detection_config(
    exp_name: str,
    use_vsaa: bool = False,
    use_asym_shrink: bool = False,
    direction_module: str | None = None,
) -> dict:
    if direction_module is None:
        direction_module = "vsaa" if use_vsaa else "none"
    direction_module = str(direction_module).strip().lower()
    use_vsaa = direction_module == "vsaa"
    return {
        "experiment": {
            "protocol_version": "paper_v2_20261109",
            "name": exp_name,
            "task": "detection",
            "paths_config": PATHS_CONFIG,
        },
        "runtime": {
            "device": "cuda",
            "allow_cpu": False,
        },
        "data": {
            "target_height": 1056,
            "target_width": 768,
            "batch_size": 16,
            "num_workers": 8,
            "shrink_ratio": 0.4,
            "augmentation": {"enabled": True, "profile": "paper_table3"},
            "label_generator": {
                "use_asymmetric_shrink": use_asym_shrink,
                "shrink_ratio_x": 0.65,
                "shrink_ratio_y": 0.90,
                "as_auxiliary": False,
                "as_mode": "edge_offset",
                "as_gamma": 1 / 3,
            },
        },
        "model": {
            "name": "DBNetPP",
            "backbone": {
                "name": "ResNetBackbone",
                "arch": "resnet34",
                "pretrained": False,
            },
            "neck": {
                "name": "DBFPN",
                "inner_channels": 128,
                "out_channels": 256,
                "use_asf": True,
                "use_vsaa": use_vsaa,
                "direction_module": direction_module,
                "strip_pool_sizes": [[20, 12], [12, 20]],
                "strip_reduction": 4,
                "coordinate_reduction": 32,
                "vsaa_reduction": 4,
                "vsaa_implementation": "paper",
                "vsaa_vertical_kernel": 15,
                "vsaa_horizontal_kernel": 5,
                "vsaa_dropout": 0.0,
                "vsaa_residual_scale": 0.1,
            },
            "head": {
                "name": "DBHead",
                "in_channels": 256,
                "k": 50,
            },
        },
        "loss": {
            "alpha": 1.0,
            "beta": 10.0,
            "gamma_as": 0.0,
            "negative_ratio": 3.0,
            "eps": 0.000001,
        },
        "eval": {
            "binary_thresh": 0.3,
            "box_thresh": 0.5,
            "unclip_ratio": 1.5,
            "iou_thresh": 0.75,
            "min_size": 3,
        },
        "train": {
            "epochs": 100,
            "lr": 0.0001,
            "min_lr": 0.000001,
            "weight_decay": 0.0001,
            "amp": True,
            "seed": 42,
            "log_interval": 20,
            "val_interval": 1,
            "grad_accum_steps": 1,
            "warmup_epochs": 5,
            "scheduler": "cosine",
        },
    }


def main() -> None:
    # Recognition formal configs
    write_yaml(
        "configs/recognition/svtr_official_baseline.yaml",
        recognition_config("svtr_official_baseline", use_dab=False, use_lortho=False),
    )

    write_yaml(
        "configs/recognition/svtr_official_dab.yaml",
        recognition_config("svtr_official_dab", use_dab=True, use_lortho=False),
    )

    write_yaml(
        "configs/recognition/svtr_official_lortho.yaml",
        recognition_config("svtr_official_lortho", use_dab=False, use_lortho=True),
    )

    write_yaml(
        "configs/recognition/svtr_official_dab_lortho.yaml",
        recognition_config("svtr_official_dab_lortho", use_dab=True, use_lortho=True),
    )

    # Section 3.3: DAB is fixed and only the constant OTP weight changes.
    for exp_name, lambda_value in [
        ("svtr_dab_otp_lambda_0", 0.0),
        ("svtr_dab_otp_lambda_0p025", 0.025),
        ("svtr_dab_otp_lambda_0p05", 0.05),
        ("svtr_dab_otp_lambda_0p1", 0.1),
        ("svtr_dab_otp_lambda_0p2", 0.2),
        ("svtr_dab_otp_lambda_0p5", 0.5),
    ]:
        write_yaml(
            f"configs/recognition/{exp_name}.yaml",
            otp_lambda_sensitivity_config(exp_name, lambda_value),
        )

    # Detection formal configs
    write_yaml(
        "configs/detection/dbnetpp_official_baseline.yaml",
        detection_config("dbnetpp_official_baseline", use_vsaa=False, use_asym_shrink=False),
    )

    write_yaml(
        "configs/detection/dbnetpp_vsaa.yaml",
        detection_config("dbnetpp_vsaa", use_vsaa=True, use_asym_shrink=False),
    )

    write_yaml(
        "configs/detection/dbnetpp_strip_pooling.yaml",
        detection_config(
            "dbnetpp_strip_pooling",
            direction_module="strip_pooling",
        ),
    )

    write_yaml(
        "configs/detection/dbnetpp_coordinate_attention.yaml",
        detection_config(
            "dbnetpp_coordinate_attention",
            direction_module="coordinate_attention",
        ),
    )

    write_yaml(
        "configs/detection/dbnetpp_horizontal_strip.yaml",
        detection_config(
            "dbnetpp_horizontal_strip",
            direction_module="horizontal_strip",
        ),
    )

    write_yaml(
        "configs/detection/dbnetpp_vertical_strip.yaml",
        detection_config(
            "dbnetpp_vertical_strip",
            direction_module="vertical_strip",
        ),
    )

    write_yaml(
        "configs/detection/dbnetpp_asym_shrink.yaml",
        detection_config("dbnetpp_asym_shrink", use_vsaa=False, use_asym_shrink=True),
    )

    write_yaml(
        "configs/detection/dbnetpp_vsaa_asym_shrink.yaml",
        detection_config("dbnetpp_vsaa_asym_shrink", use_vsaa=True, use_asym_shrink=True),
    )


if __name__ == "__main__":
    main()
