import argparse
import copy
import os
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader

from manchu_ocr.data.collate.rec_collate import RecCollate
from manchu_ocr.data.datasets.recognition_dataset import RecognitionDataset
from manchu_ocr.data.label_converters.ctc_label_converter import CTCLabelConverter
from manchu_ocr.data.transforms.rec_transforms import build_rec_transform
from manchu_ocr.losses.ctc_loss import CTCLossWrapper
from manchu_ocr.losses.orthographic_transition_loss import OrthographicTransitionLoss
from manchu_ocr.losses.sequence_alignment_loss import SequenceAlignmentLoss
from manchu_ocr.metrics.recognition_metrics import (
    compute_recognition_metrics,
    levenshtein_distance,
)
from manchu_ocr.models.recognition.builder import build_recognition_model
from manchu_ocr.models.recognition.decoders.lexicon_corrector import (
    CTCLexiconReranker,
    LexiconCorrector,
    load_lexicon_from_manifest,
)
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.experiment_result import (
    ExperimentTimer,
    build_experiment_result,
    collect_torch_runtime,
    format_experiment_result,
)
from manchu_ocr.utils.file_io import save_json
from manchu_ocr.utils.logger import setup_logger
from manchu_ocr.utils.seed import set_seed


def build_dataloader(
    manifest_path: str,
    charset_path: str,
    image_height: int,
    image_width: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, CTCLabelConverter]:
    converter = CTCLabelConverter(charset_path)

    transform = build_rec_transform(
        image_height=image_height,
        image_width=image_width,
        keep_aspect_ratio=True,
    )

    dataset = RecognitionDataset(
        manifest_path=manifest_path,
        transform=transform,
        check_exists=False,
    )

    collate_fn = RecCollate(label_converter=converter)

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4

    dataloader = DataLoader(**dataloader_kwargs)

    return dataloader, converter


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    batch["images"] = batch["images"].to(device, non_blocking=True)
    batch["targets"] = batch["targets"].to(device, non_blocking=True)
    batch["target_lengths"] = batch["target_lengths"].to(device, non_blocking=True)
    return batch


def build_model(cfg: dict, num_classes: int):
    return build_recognition_model(
        cfg=cfg,
        num_classes=num_classes,
    )


def build_lexicon_corrector(cfg: Dict, train_list: str, logger) -> Optional[LexiconCorrector]:
    decode_cfg = cfg.get("decode", {})

    if not bool(decode_cfg.get("use_lexicon", False)):
        return None

    lexicon_source = decode_cfg.get("lexicon_source", "train")

    if lexicon_source != "train":
        raise ValueError(
            f"Unsupported lexicon_source={lexicon_source}. "
            "Use train to avoid validation/test leakage."
        )

    lexicon_counts = load_lexicon_from_manifest(train_list)
    corrector = LexiconCorrector(
        lexicon_counts=lexicon_counts,
        max_edit_distance=int(decode_cfg.get("max_edit_distance", 2)),
        length_delta=int(decode_cfg.get("length_delta", 2)),
        min_word_length=int(decode_cfg.get("min_word_length", 2)),
    )

    logger.info(
        f"Lexicon correction enabled. words={len(lexicon_counts)}, "
        f"max_edit_distance={corrector.max_edit_distance}, "
        f"length_delta={corrector.length_delta}, "
        f"min_word_length={corrector.min_word_length}"
    )

    return corrector


def build_ctc_lexicon_reranker(
    cfg: Dict,
    train_list: str,
    converter: CTCLabelConverter,
    logger,
) -> Optional[CTCLexiconReranker]:
    decode_cfg = cfg.get("decode", {})

    if not bool(decode_cfg.get("use_ctc_lexicon_rerank", False)):
        return None

    lexicon_source = decode_cfg.get("lexicon_source", "train")

    if lexicon_source != "train":
        raise ValueError(
            f"Unsupported lexicon_source={lexicon_source}. "
            "Use train to avoid validation/test leakage."
        )

    lexicon_counts = load_lexicon_from_manifest(train_list)
    reranker = CTCLexiconReranker(
        lexicon_counts=lexicon_counts,
        char_to_idx=converter.char_to_idx,
        blank_idx=converter.blank_idx,
        max_edit_distance=int(decode_cfg.get("ctc_rerank_max_edit_distance", 2)),
        length_delta=int(decode_cfg.get("ctc_rerank_length_delta", 2)),
        min_word_length=int(decode_cfg.get("ctc_rerank_min_word_length", 2)),
        max_candidates=int(decode_cfg.get("ctc_rerank_max_candidates", 40)),
        prior_weight=float(decode_cfg.get("ctc_rerank_prior_weight", 0.05)),
    )

    logger.info(
        f"CTC lexicon rerank enabled. words={len(lexicon_counts)}, "
        f"max_edit_distance={reranker.max_edit_distance}, "
        f"length_delta={reranker.length_delta}, "
        f"max_candidates={reranker.max_candidates}, "
        f"prior_weight={reranker.prior_weight}"
    )

    return reranker


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: CTCLossWrapper,
    converter: CTCLabelConverter,
    device: torch.device,
    max_batches: Optional[int] = None,
    ortho_criterion: Optional[OrthographicTransitionLoss] = None,
    lambda_ortho: float = 0.0,
    align_criterion: Optional[SequenceAlignmentLoss] = None,
    lambda_align: float = 0.0,
    lexicon_corrector: Optional[LexiconCorrector] = None,
    ctc_lexicon_reranker: Optional[CTCLexiconReranker] = None,
) -> tuple[Dict[str, float], list[Dict]]:
    model.eval()

    total_loss = 0.0
    total_ctc_loss = 0.0
    total_ortho_loss = 0.0
    total_align_loss = 0.0
    total_batches = 0

    all_raw_preds = []
    all_preds = []
    all_labels = []
    all_records = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        logits = model(batch["images"])
        logits_for_loss = logits.float()

        ctc_loss = criterion(
            logits=logits_for_loss,
            targets=batch["targets"],
            target_lengths=batch["target_lengths"],
        )

        if ortho_criterion is not None and lambda_ortho > 0:
            ortho_loss = ortho_criterion(logits_for_loss)
        else:
            ortho_loss = logits_for_loss.new_tensor(0.0)

        if align_criterion is not None and lambda_align > 0:
            align_loss = align_criterion(
                logits=logits_for_loss,
                targets=batch["targets"],
                target_lengths=batch["target_lengths"],
            )
        else:
            align_loss = logits_for_loss.new_tensor(0.0)

        loss = ctc_loss + lambda_ortho * ortho_loss + lambda_align * align_loss

        raw_preds = converter.decode_logits(logits_for_loss.detach().cpu())
        if ctc_lexicon_reranker is not None:
            preds = ctc_lexicon_reranker.rerank_batch(logits_for_loss, raw_preds)
        elif lexicon_corrector is not None:
            preds = lexicon_corrector.correct_batch(raw_preds)
        else:
            preds = raw_preds

        labels = batch["labels"]
        image_paths = batch["image_paths"]

        all_raw_preds.extend(raw_preds)
        all_preds.extend(preds)
        all_labels.extend(labels)

        for image_path, raw_pred, pred, label in zip(image_paths, raw_preds, preds, labels):
            edit_distance = levenshtein_distance(pred, label)
            raw_edit_distance = levenshtein_distance(raw_pred, label)
            all_records.append(
                {
                    "image_path": image_path,
                    "raw_prediction": raw_pred,
                    "prediction": pred,
                    "label": label,
                    "correct": pred == label,
                    "exact_correct": pred == label,
                    "raw_correct": raw_pred == label,
                    "raw_exact_correct": raw_pred == label,
                    "first_char_match": (
                        len(pred) > 0 and len(label) > 0 and pred[0] == label[0]
                    ),
                    "second_char_match": (
                        len(label) < 2
                        or (len(pred) > 1 and pred[1] == label[1])
                    ),
                    "last_char_match": (
                        len(pred) > 0 and len(label) > 0 and pred[-1] == label[-1]
                    ),
                    "raw_first_char_match": (
                        len(raw_pred) > 0 and len(label) > 0 and raw_pred[0] == label[0]
                    ),
                    "raw_second_char_match": (
                        len(label) < 2
                        or (len(raw_pred) > 1 and raw_pred[1] == label[1])
                    ),
                    "raw_last_char_match": (
                        len(raw_pred) > 0 and len(label) > 0 and raw_pred[-1] == label[-1]
                    ),
                    "edit_distance": edit_distance,
                    "raw_edit_distance": raw_edit_distance,
                    "lexicon_changed": raw_pred != pred,
                }
            )

        total_loss += loss.item()
        total_ctc_loss += ctc_loss.item()
        total_ortho_loss += ortho_loss.item()
        total_align_loss += align_loss.item()
        total_batches += 1

    metrics = compute_recognition_metrics(all_preds, all_labels)

    metrics["loss"] = total_loss / max(total_batches, 1)
    metrics["ctc_loss"] = total_ctc_loss / max(total_batches, 1)
    metrics["ortho_loss"] = total_ortho_loss / max(total_batches, 1)
    metrics["align_loss"] = total_align_loss / max(total_batches, 1)
    metrics["num_samples"] = len(all_labels)
    metrics["num_batches"] = total_batches

    return metrics, all_records


def main() -> None:
    run_timer = ExperimentTimer()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Recognition config yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--disable-lexicon", action="store_true")
    parser.add_argument("--enable-ctc-rerank", action="store_true")
    parser.add_argument("--disable-ctc-rerank", action="store_true")
    parser.add_argument("--decode-max-edit-distance", type=int, default=None)
    parser.add_argument("--decode-length-delta", type=int, default=None)
    parser.add_argument("--ctc-rerank-max-edit-distance", type=int, default=None)
    parser.add_argument("--ctc-rerank-length-delta", type=int, default=None)
    parser.add_argument("--ctc-rerank-max-candidates", type=int, default=None)
    parser.add_argument("--ctc-rerank-prior-weight", type=float, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths_cfg = load_yaml(cfg["experiment"]["paths_config"])

    decode_cfg = cfg.setdefault("decode", {})

    if args.disable_lexicon:
        decode_cfg["use_lexicon"] = False
        decode_cfg["use_ctc_lexicon_rerank"] = False

    if args.enable_ctc_rerank:
        decode_cfg["use_lexicon"] = True
        decode_cfg["use_ctc_lexicon_rerank"] = True

    if args.disable_ctc_rerank:
        decode_cfg["use_ctc_lexicon_rerank"] = False

    if args.decode_max_edit_distance is not None:
        decode_cfg["max_edit_distance"] = args.decode_max_edit_distance

    if args.decode_length_delta is not None:
        decode_cfg["length_delta"] = args.decode_length_delta

    if args.ctc_rerank_max_edit_distance is not None:
        decode_cfg["ctc_rerank_max_edit_distance"] = args.ctc_rerank_max_edit_distance

    if args.ctc_rerank_length_delta is not None:
        decode_cfg["ctc_rerank_length_delta"] = args.ctc_rerank_length_delta

    if args.ctc_rerank_max_candidates is not None:
        decode_cfg["ctc_rerank_max_candidates"] = args.ctc_rerank_max_candidates

    if args.ctc_rerank_prior_weight is not None:
        decode_cfg["ctc_rerank_prior_weight"] = args.ctc_rerank_prior_weight

    set_seed(cfg["train"].get("seed", 42))

    exp_name = cfg["experiment"]["name"]
    logger = setup_logger(f"eval_{exp_name}_{args.split}")

    runtime_cfg = cfg.get("runtime", {})
    requested_device = runtime_cfg.get("device", "auto")
    allow_cpu = bool(runtime_cfg.get("allow_cpu", True))

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config requires CUDA, but torch.cuda.is_available() is False."
            )
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "Evaluation is running on CPU, but allow_cpu=false."
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rec_cfg = paths_cfg["recognition_data"]

    if args.split == "train":
        manifest_path = rec_cfg["train_list"]
    elif args.split == "val":
        manifest_path = rec_cfg["val_list"]
    else:
        manifest_path = rec_cfg["test_list"]

    charset_path = rec_cfg["charset"]
    train_list = rec_cfg["train_list"]

    batch_size = args.batch_size or int(cfg["data"]["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["data"].get("num_workers", 0))
    image_height = int(cfg["data"]["image_height"])
    image_width = int(cfg["data"]["image_width"])

    output_root = Path(paths_cfg["outputs"]["root"])
    metrics_dir = output_root / "metrics" / "recognition" / exp_name
    pred_dir = output_root / "predictions" / "recognition" / exp_name

    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(
        "Recognition metric protocol: strict exact-match WA; "
        "CA=(N-S-D)/N; CER=(S+D+I)/N."
    )

    dataloader, converter = build_dataloader(
        manifest_path=manifest_path,
        charset_path=charset_path,
        image_height=image_height,
        image_width=image_width,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logger.info(f"Dataset samples: {len(dataloader.dataset)}")
    logger.info(f"Batches: {len(dataloader)}")
    logger.info(f"CTC num_classes: {converter.num_classes}")

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain key 'model': {checkpoint_path}")

    checkpoint_cfg = checkpoint.get("config")
    model_build_cfg = cfg

    if isinstance(checkpoint_cfg, dict) and isinstance(checkpoint_cfg.get("model"), dict):
        model_build_cfg = copy.deepcopy(cfg)
        model_build_cfg["model"] = copy.deepcopy(checkpoint_cfg["model"])

        if checkpoint_cfg.get("model") != cfg.get("model"):
            logger.warning(
                "Current YAML model section differs from checkpoint config. "
                "Using checkpoint model config for checkpoint-compatible evaluation."
            )
        else:
            logger.info("Using checkpoint model config for evaluation.")
    else:
        logger.warning(
            "Checkpoint has no saved model config. Falling back to current YAML model config."
        )

    ctc_lexicon_reranker = build_ctc_lexicon_reranker(
        cfg=cfg,
        train_list=train_list,
        converter=converter,
        logger=logger,
    )

    model = build_model(model_build_cfg, num_classes=converter.num_classes).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    logger.info(
        f"Loaded checkpoint epoch={checkpoint.get('epoch', 'unknown')}, "
        f"best_word_accuracy={checkpoint.get('best_word_accuracy', 'unknown')}"
    )

    criterion = CTCLossWrapper(
        blank_idx=int(cfg["loss"].get("blank_idx", 0)),
        zero_infinity=bool(cfg["loss"].get("zero_infinity", True)),
    )

    if ctc_lexicon_reranker is None:
        lexicon_corrector = build_lexicon_corrector(
            cfg=cfg,
            train_list=train_list,
            logger=logger,
        )
    else:
        lexicon_corrector = None
        logger.info("Skipping nearest-word lexicon correction because CTC rerank is enabled.")

    use_alignment_loss = bool(cfg["loss"].get("use_alignment_loss", False))
    lambda_align = float(cfg["loss"].get("lambda_align", 0.0))

    if use_alignment_loss and lambda_align > 0:
        align_criterion = SequenceAlignmentLoss(
            blank_idx=int(cfg["loss"].get("blank_idx", 0)),
            label_smoothing=float(cfg["loss"].get("align_label_smoothing", 0.05)),
        ).to(device)
        logger.info(
            f"Alignment loss enabled. lambda_align={lambda_align}, "
            f"label_smoothing={cfg['loss'].get('align_label_smoothing', 0.05)}"
        )
    else:
        align_criterion = None
        lambda_align = 0.0

    use_orthographic_loss = bool(cfg["loss"].get("use_orthographic_loss", False))
    lambda_ortho = float(cfg["loss"].get("lambda_ortho", 0.0))

    if use_orthographic_loss:
        ortho_criterion = OrthographicTransitionLoss(
            transition_matrix_path=cfg["loss"]["transition_matrix"],
            blank_idx=int(cfg["loss"].get("blank_idx", 0)),
            normalize_char_probs=bool(cfg["loss"].get("normalize_char_probs", True)),
            confidence_weighting=bool(cfg["loss"].get("confidence_weighting", True)),
            confidence_power=float(cfg["loss"].get("confidence_power", 1.0)),
            min_confidence=float(cfg["loss"].get("min_confidence", 0.0)),
        ).to(device)

        logger.info(f"Orthographic loss enabled. lambda_ortho={lambda_ortho}")
    else:
        ortho_criterion = None
        lambda_ortho = 0.0

    metrics, records = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        converter=converter,
        device=device,
        max_batches=args.max_batches,
        ortho_criterion=ortho_criterion,
        lambda_ortho=lambda_ortho,
        align_criterion=align_criterion,
        lambda_align=lambda_align,
        lexicon_corrector=lexicon_corrector,
        ctc_lexicon_reranker=ctc_lexicon_reranker,
    )

    wrong_records = [r for r in records if not r["correct"]]

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timing = run_timer.finish()
    runtime_details = collect_torch_runtime(device)
    runtime_details.update(
        {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "num_samples": int(metrics["num_samples"]),
            "num_batches": int(metrics["num_batches"]),
            "samples_per_second": round(
                float(metrics["num_samples"]) / max(timing.elapsed_seconds, 1e-12),
                6,
            ),
        }
    )

    model_architecture = str(
        model_build_cfg.get("model", {}).get("name", model.__class__.__name__)
    )
    result = build_experiment_result(
        task="recognition",
        experiment_name=exp_name,
        model_name=exp_name,
        model_architecture=model_architecture,
        split=args.split,
        config_path=args.config,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        metrics=metrics,
        timing=timing,
        runtime_details=runtime_details,
        extra_fields={
            "experiment": exp_name,
            "legacy_split": args.split,
            "checkpoint": str(checkpoint_path).replace("\\", "/"),
            "checkpoint_epoch": checkpoint.get("epoch", None),
            "checkpoint_best_word_accuracy": checkpoint.get(
                "best_word_accuracy",
                None,
            ),
            "evaluation_protocol": {
                "strict": True,
                "word_accuracy": "exact_match",
                "character_accuracy": "(N-S-D)/N",
                "cer": "(S+D+I)/N",
            },
            "num_wrong": len(wrong_records),
        },
    )

    suffix = args.output_suffix
    save_json(result, metrics_dir / f"eval_{args.split}{suffix}.json")

    if args.save_predictions:
        save_json(records, pred_dir / f"{args.split}{suffix}_predictions.json")
        save_json(wrong_records[:1000], pred_dir / f"{args.split}{suffix}_wrong_first_1000.json")

    logger.info(
        f"Evaluation finished. "
        f"loss={metrics['loss']:.4f}, "
        f"WA={metrics['word_accuracy']:.4f}, "
        f"CA={metrics['character_accuracy']:.4f}, "
        f"CER={metrics['cer']:.4f}, "
        f"EditDist={metrics['edit_distance']:.4f}, "
        f"Samples={metrics['num_samples']}"
    )
    logger.info(format_experiment_result(result))

    logger.info(f"Saved metrics to: {metrics_dir / f'eval_{args.split}{suffix}.json'}")


if __name__ == "__main__":
    main()
