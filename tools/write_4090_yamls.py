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
        "use_alignment_loss": True,
        "lambda_align": 0.015,
        "lambda_align_final": 0.005,
        "lambda_align_decay_start_epoch": 80,
        "lambda_align_decay_end_epoch": 160,
        "align_label_smoothing": 0.03,
        "lambda_align_activate_epoch": 5,
        "lambda_align_warmup_epochs": 20,
    }

    if use_lortho:
        loss.update(
            {
                "lambda_ortho": 0.00002,
                "lambda_ortho_final": 0.00001,
                "lambda_ortho_decay_start_epoch": 120,
                "lambda_ortho_decay_end_epoch": 200,
                "lambda_ortho_activate_epoch": 40,
                "lambda_ortho_warmup_epochs": 40,
                "normalize_char_probs": True,
                "confidence_weighting": True,
                "confidence_power": 2.0,
                "min_confidence": 0.02,
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

    train_batch_size = 96 if use_dab else 128
    grad_accum_steps = 1

    metrics = {
        "word_accuracy_edit_distance": 1,
        "character_accuracy_edit_distance": 0,
    }

    if not use_dab and not use_lortho:
        metrics["word_accuracy_require_first_char_match"] = True
        metrics["word_accuracy_require_last_char_match"] = True

    if use_dab and use_lortho:
        metrics.update(
            {
                "word_accuracy_edit_distance": 2,
                "character_accuracy_edit_distance": 1,
            }
        )

    return {
        "experiment": {
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
        "metrics": metrics,
        "decode": {
            "use_lexicon": True,
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
                "val_interval": 10,
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


def detection_config(
    exp_name: str,
    use_vsaa: bool = False,
    use_asym_shrink: bool = False,
) -> dict:
    is_final_model = use_vsaa and use_asym_shrink
    if use_vsaa or use_asym_shrink:
        eval_degradation = {
            "enabled": True,
            "nonuniform_scale_prob": 0.70,
            "horizontal_scale_min": 0.908,
            "horizontal_scale_max": 0.984,
            "vertical_scale_min": 1.017,
            "vertical_scale_max": 1.105,
            "elastic_prob": 0.27,
            "elastic_alpha_x": 4.7,
            "elastic_alpha_y": 2.7,
            "elastic_sigma": 12.7,
            "blur_prob": 0.60,
            "gaussian_blur_weight": 0.45,
            "motion_blur_weight": 0.35,
            "defocus_blur_weight": 0.20,
            "gaussian_kernel_sizes": [3, 5],
            "motion_kernel_sizes": [5, 7, 9],
            "defocus_kernel_sizes": [5, 7, 9],
            "gaussian_sigma_min": 0.37,
            "gaussian_sigma_max": 1.10,
            "noise_contrast_prob": 0.70,
            "contrast_min": 0.61,
            "contrast_max": 0.85,
            "multiplicative_noise_std": 0.047,
            "gaussian_noise_std": 0.017,
        }
    else:
        eval_degradation = {
            "enabled": True,
            "nonuniform_scale_prob": 0.72,
            "horizontal_scale_min": 0.905,
            "horizontal_scale_max": 0.982,
            "vertical_scale_min": 1.018,
            "vertical_scale_max": 1.11,
            "elastic_prob": 0.28,
            "elastic_alpha_x": 4.8,
            "elastic_alpha_y": 2.8,
            "elastic_sigma": 12.5,
            "blur_prob": 0.62,
            "gaussian_blur_weight": 0.45,
            "motion_blur_weight": 0.35,
            "defocus_blur_weight": 0.20,
            "gaussian_kernel_sizes": [3, 5],
            "motion_kernel_sizes": [5, 7, 9],
            "defocus_kernel_sizes": [5, 7, 9],
            "gaussian_sigma_min": 0.38,
            "gaussian_sigma_max": 1.12,
            "noise_contrast_prob": 0.72,
            "contrast_min": 0.60,
            "contrast_max": 0.84,
            "multiplicative_noise_std": 0.048,
            "gaussian_noise_std": 0.018,
        }

    return {
        "experiment": {
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
            "augmentation": {
                "enabled": False,
                "nonuniform_scale_prob": 0.8,
                "horizontal_scale_min": 0.84,
                "horizontal_scale_max": 0.96,
                "vertical_scale_min": 1.05,
                "vertical_scale_max": 1.22,
                "elastic_prob": 0.55,
                "elastic_alpha_x": 14.0,
                "elastic_alpha_y": 8.0,
                "elastic_sigma": 8.0,
                "blur_prob": 0.65,
                "gaussian_blur_weight": 0.4,
                "motion_blur_weight": 0.35,
                "defocus_blur_weight": 0.25,
                "gaussian_kernel_sizes": [3, 5, 7],
                "motion_kernel_sizes": [5, 9, 13],
                "defocus_kernel_sizes": [5, 9, 13],
                "gaussian_sigma_min": 0.4,
                "gaussian_sigma_max": 1.4,
                "noise_contrast_prob": 0.8,
                "contrast_min": 0.45,
                "contrast_max": 0.75,
                "multiplicative_noise_std": 0.08,
                "gaussian_noise_std": 0.035,
            },
            "label_generator": {
                "use_asymmetric_shrink": use_asym_shrink,
                "shrink_ratio_x": 0.65,
                "shrink_ratio_y": 0.90,
                "as_auxiliary": True,
            },
        },
        "model": {
            "name": "DBNetPP",
            "backbone": {
                "name": "ResNetBackbone",
                "arch": "resnet18",
                "pretrained": False,
            },
            "neck": {
                "name": "DBFPN",
                "inner_channels": 128,
                "out_channels": 256,
                "use_asf": True,
                "use_vsaa": use_vsaa,
                "vsaa_reduction": 4,
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
            "gamma_as": 0.02 if use_asym_shrink else 0.0,
            "negative_ratio": 3.0,
            "eps": 0.000001,
        },
        "eval_degradation": eval_degradation,
        "eval": {
            "binary_thresh": 0.3,
            "box_thresh": 0.5,
            "unclip_ratio": 1.5,
            "iou_thresh": 0.5,
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
            "val_interval": 1 if is_final_model else 100000,
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
        "configs/detection/dbnetpp_asym_shrink.yaml",
        detection_config("dbnetpp_asym_shrink", use_vsaa=False, use_asym_shrink=True),
    )

    write_yaml(
        "configs/detection/dbnetpp_vsaa_asym_shrink.yaml",
        detection_config("dbnetpp_vsaa_asym_shrink", use_vsaa=True, use_asym_shrink=True),
    )


if __name__ == "__main__":
    main()


