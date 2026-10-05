import argparse
import copy
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from manchu_ocr.data.collate.rec_collate import RecCollate
from manchu_ocr.data.datasets.recognition_dataset import RecognitionDataset
from manchu_ocr.data.label_converters.ctc_label_converter import CTCLabelConverter
from manchu_ocr.data.transforms.rec_transforms import build_rec_transform
from manchu_ocr.losses.ctc_loss import CTCLossWrapper
from manchu_ocr.losses.nrtr_loss import NRTRLoss
from manchu_ocr.losses.orthographic_transition_loss import OrthographicTransitionLoss
from manchu_ocr.losses.sequence_alignment_loss import SequenceAlignmentLoss
from manchu_ocr.metrics.recognition_metrics import compute_recognition_metrics
from manchu_ocr.models.recognition.builder import build_recognition_model
from manchu_ocr.models.recognition.decoders.lexicon_corrector import (
    CTCLexiconReranker,
    LexiconCorrector,
    load_lexicon_from_manifest,
)
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import save_json
from manchu_ocr.utils.logger import setup_logger
from manchu_ocr.utils.seed import set_seed

def update_ema_model(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
) -> None:
    with torch.no_grad():
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()

        for key, ema_value in ema_state.items():
            model_value = model_state[key]

            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(model_value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


def scheduled_weight(
    initial: float,
    final: float,
    start_epoch: int,
    end_epoch: int,
    epoch: int,
) -> float:
    if initial <= 0:
        return 0.0

    if end_epoch <= start_epoch:
        return float(final)

    if epoch <= start_epoch:
        return float(initial)

    if epoch >= end_epoch:
        return float(final)

    progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
    return float(initial + (final - initial) * progress)


def apply_loss_warmup(
    weight: float,
    epoch: int,
    activate_epoch: int = 1,
    warmup_epochs: int = 0,
) -> float:
    """
    Keep auxiliary objectives from dominating early CTC alignment learning.
    """
    if weight <= 0:
        return 0.0

    activate_epoch = max(1, int(activate_epoch))
    warmup_epochs = max(0, int(warmup_epochs))

    if epoch < activate_epoch:
        return 0.0

    if warmup_epochs > 0:
        progress = (epoch - activate_epoch + 1) / float(warmup_epochs)
        return float(weight) * min(max(progress, 0.0), 1.0)

    return float(weight)


def build_dataloader(
    manifest_path: str,
    charset_path: str,
    image_height: int,
    image_width: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: bool = False,
    augmentation_cfg: Optional[Dict] = None,
) -> tuple[DataLoader, CTCLabelConverter]:
    converter = CTCLabelConverter(charset_path)

    transform = build_rec_transform(
        image_height=image_height,
        image_width=image_width,
        keep_aspect_ratio=True,
        augment=augment,
        augmentation_cfg=augmentation_cfg,
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
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": shuffle,
    }

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4

    dataloader = DataLoader(**dataloader_kwargs)

    return dataloader, converter


def move_batch_to_device(
    batch: Dict,
    device: torch.device,
    channels_last: bool = False,
) -> Dict:
    batch["images"] = batch["images"].to(device, non_blocking=True)
    if channels_last:
        batch["images"] = batch["images"].contiguous(memory_format=torch.channels_last)
    batch["targets"] = batch["targets"].to(device, non_blocking=True)
    batch["target_lengths"] = batch["target_lengths"].to(device, non_blocking=True)
    return batch


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
        f"CTC lexicon rerank available for periodic validation. "
        f"words={len(lexicon_counts)}, "
        f"max_edit_distance={reranker.max_edit_distance}, "
        f"length_delta={reranker.length_delta}, "
        f"max_candidates={reranker.max_candidates}, "
        f"prior_weight={reranker.prior_weight}"
    )

    return reranker


@torch.no_grad()
def validate(
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
    channels_last: bool = False,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_ctc_loss = 0.0
    total_ortho_loss = 0.0
    total_align_loss = 0.0
    total_batches = 0

    all_preds = []
    all_ctc_rerank_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device, channels_last=channels_last)

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
        preds = (
            lexicon_corrector.correct_batch(raw_preds)
            if lexicon_corrector is not None
            else raw_preds
        )
        ctc_rerank_preds = (
            ctc_lexicon_reranker.rerank_batch(logits_for_loss, raw_preds)
            if ctc_lexicon_reranker is not None
            else None
        )

        all_preds.extend(preds)
        if ctc_rerank_preds is not None:
            all_ctc_rerank_preds.extend(ctc_rerank_preds)
        all_labels.extend(batch["labels"])

        total_loss += loss.item()
        total_ctc_loss += ctc_loss.item()
        total_ortho_loss += ortho_loss.item()
        total_align_loss += align_loss.item()
        total_batches += 1

    metrics = compute_recognition_metrics(all_preds, all_labels)

    if ctc_lexicon_reranker is not None:
        ctc_rerank_metrics = compute_recognition_metrics(
            all_ctc_rerank_preds,
            all_labels,
        )
        metrics["ctc_rerank_word_accuracy"] = ctc_rerank_metrics["word_accuracy"]
        metrics["ctc_rerank_character_accuracy"] = ctc_rerank_metrics["character_accuracy"]
        metrics["ctc_rerank_cer"] = ctc_rerank_metrics["cer"]
        metrics["ctc_rerank_edit_distance"] = ctc_rerank_metrics["edit_distance"]
        metrics["ctc_rerank_enabled"] = True
    else:
        metrics["ctc_rerank_word_accuracy"] = float("nan")
        metrics["ctc_rerank_character_accuracy"] = float("nan")
        metrics["ctc_rerank_cer"] = float("nan")
        metrics["ctc_rerank_edit_distance"] = float("nan")
        metrics["ctc_rerank_enabled"] = False

    metrics["loss"] = total_loss / max(total_batches, 1)
    metrics["ctc_loss"] = total_ctc_loss / max(total_batches, 1)
    metrics["ortho_loss"] = total_ortho_loss / max(total_batches, 1)
    metrics["align_loss"] = total_align_loss / max(total_batches, 1)

    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_word_accuracy: float,
    cfg: Dict,
    best_metric_name: str = "word_accuracy",
    best_metric_value: float = -1.0,
    ema_model: Optional[torch.nn.Module] = None,
    scheduler: Optional[object] = None,
    scaler: Optional[GradScaler] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    train_model_state = model.state_dict()
    eval_model_state = ema_model.state_dict() if ema_model is not None else train_model_state

    state = {
        "epoch": epoch,
        # Keep "model" as the evaluation weight for backward-compatible eval scripts.
        "model": eval_model_state,
        # Keep raw training weights separately so resume remains optimizer-consistent.
        "train_model": train_model_state,
        "ema_model": eval_model_state if ema_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "best_word_accuracy": best_word_accuracy,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "config": cfg,
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/recognition/svtr_baseline.yaml",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint path.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from outputs/checkpoints/.../last.pth if it exists.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths_cfg = load_yaml(cfg["experiment"]["paths_config"])

    exp_name = cfg["experiment"]["name"]

    set_seed(cfg["train"].get("seed", 42))

    output_root = Path(paths_cfg["outputs"]["root"])
    log_dir = output_root / "logs" / "recognition" / exp_name
    ckpt_dir = output_root / "checkpoints" / "recognition" / exp_name
    metrics_dir = output_root / "metrics" / "recognition" / exp_name

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name=f"train_{exp_name}",
        log_file=log_dir / "train.log",
    )

    shutil.copyfile(args.config, log_dir / "config.yaml")

    rec_cfg = paths_cfg["recognition_data"]

    train_list = rec_cfg["train_list"]
    val_list = rec_cfg["val_list"]
    charset_path = rec_cfg["charset"]

    runtime_cfg = cfg.get("runtime", {})
    requested_device = runtime_cfg.get("device", "auto")
    allow_cpu = bool(runtime_cfg.get("allow_cpu", True))

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config requires CUDA, but torch.cuda.is_available() is False. "
                "Please install CUDA-enabled PyTorch in the current conda environment."
            )
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "Training is running on CPU, but allow_cpu=false. "
            "Please fix CUDA/PyTorch installation before training."
        )

    train_runtime_cfg = cfg.get("train", {})
    validate_during_training = bool(
        train_runtime_cfg.get("validate_during_training", True)
    )
    use_channels_last = bool(train_runtime_cfg.get("channels_last", False)) and device.type == "cuda"

    if bool(train_runtime_cfg.get("cudnn_benchmark", True)) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if bool(train_runtime_cfg.get("allow_tf32", True)) and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Train list: {train_list}")
    logger.info(f"Val list: {val_list}")
    logger.info(f"Charset: {charset_path}")
    logger.info(
        f"Runtime speed options: channels_last={use_channels_last}, "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark}, "
        f"allow_tf32={torch.backends.cuda.matmul.allow_tf32 if device.type == 'cuda' else False}"
    )
    logger.info(f"Validation during training: {validate_during_training}")

    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    image_height = int(cfg["data"]["image_height"])
    image_width = int(cfg["data"]["image_width"])
    augmentation_cfg = cfg["data"].get("augmentation", {})
    use_train_augmentation = bool(augmentation_cfg.get("enabled", False))

    if args.batch_size is not None:
        batch_size = int(args.batch_size)

    if args.num_workers is not None:
        num_workers = int(args.num_workers)

    if args.debug:
        batch_size = min(batch_size, 8)
        if args.max_train_batches is None:
            args.max_train_batches = 20
        if args.max_val_batches is None:
            args.max_val_batches = 5
        logger.info("Debug mode enabled.")

    train_loader, converter = build_dataloader(
        manifest_path=train_list,
        charset_path=charset_path,
        image_height=image_height,
        image_width=image_width,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        augment=use_train_augmentation,
        augmentation_cfg=augmentation_cfg,
    )

    if validate_during_training:
        val_loader, _ = build_dataloader(
            manifest_path=val_list,
            charset_path=charset_path,
            image_height=image_height,
            image_width=image_width,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            augment=False,
            augmentation_cfg=None,
        )
    else:
        val_loader = None

    logger.info(f"CTC num_classes: {converter.num_classes}")
    logger.info(f"Train batches: {len(train_loader)}")
    if val_loader is not None:
        logger.info(f"Val batches: {len(val_loader)}")
    else:
        logger.info("Val loader skipped because validate_during_training=false.")
    logger.info(f"Train augmentation enabled: {use_train_augmentation}, cfg={augmentation_cfg}")
    logger.info(
        "Recognition metric protocol: strict exact-match WA; "
        "CA=(N-S-D)/N; CER=(S+D+I)/N."
    )

    if validate_during_training:
        lexicon_corrector = build_lexicon_corrector(
            cfg=cfg,
            train_list=train_list,
            logger=logger,
        )
    else:
        lexicon_corrector = None
        logger.info(
            "Skipping validation lexicon setup because "
            "validate_during_training=false."
        )

    model = build_recognition_model(
        cfg=cfg,
        num_classes=converter.num_classes,
    ).to(device)

    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    use_ema = bool(cfg["train"].get("use_ema", False))
    ema_decay = float(cfg["train"].get("ema_decay", 0.999))

    if use_ema:
        ema_model = copy.deepcopy(model).eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
        logger.info(f"EMA enabled. decay={ema_decay}")
    else:
        ema_model = None

    criterion = CTCLossWrapper(
        blank_idx=int(cfg["loss"].get("blank_idx", 0)),
        zero_infinity=bool(cfg["loss"].get("zero_infinity", True)),
    )

    use_nrtr_loss = bool(cfg["loss"].get("use_nrtr_loss", False))
    lambda_nrtr = float(cfg["loss"].get("lambda_nrtr", 0.0))
    lambda_nrtr_initial = lambda_nrtr
    lambda_nrtr_final = float(cfg["loss"].get("lambda_nrtr_final", lambda_nrtr))
    lambda_nrtr_decay_start = int(
        cfg["loss"].get("lambda_nrtr_decay_start_epoch", 0)
    )
    lambda_nrtr_decay_end = int(
        cfg["loss"].get("lambda_nrtr_decay_end_epoch", 0)
    )
    lambda_nrtr_activate_epoch = int(
        cfg["loss"].get("lambda_nrtr_activate_epoch", 1)
    )
    lambda_nrtr_warmup_epochs = int(
        cfg["loss"].get("lambda_nrtr_warmup_epochs", 0)
    )

    if use_nrtr_loss:
        if not callable(getattr(model, "forward_train", None)):
            raise TypeError(
                "use_nrtr_loss=true requires a recognizer with forward_train()."
            )
        nrtr_criterion = NRTRLoss(
            pad_idx=int(getattr(model, "nrtr_pad_idx", 0)),
            label_smoothing=float(cfg["loss"].get("nrtr_label_smoothing", 0.0)),
        ).to(device)
        logger.info(
            "NRTR auxiliary loss enabled. "
            f"lambda_nrtr={lambda_nrtr}, "
            f"label_smoothing={cfg['loss'].get('nrtr_label_smoothing', 0.0)}"
        )
    else:
        nrtr_criterion = None
        lambda_nrtr = 0.0
        lambda_nrtr_initial = 0.0
        lambda_nrtr_final = 0.0

    use_alignment_loss = bool(cfg["loss"].get("use_alignment_loss", False))
    lambda_align = float(cfg["loss"].get("lambda_align", 0.0))
    lambda_align_initial = lambda_align
    lambda_align_final = float(cfg["loss"].get("lambda_align_final", lambda_align))
    lambda_align_decay_start = int(cfg["loss"].get("lambda_align_decay_start_epoch", 0))
    lambda_align_decay_end = int(cfg["loss"].get("lambda_align_decay_end_epoch", 0))
    lambda_align_activate_epoch = int(cfg["loss"].get("lambda_align_activate_epoch", 1))
    lambda_align_warmup_epochs = int(cfg["loss"].get("lambda_align_warmup_epochs", 0))

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
        logger.info(f"Transition matrix: {cfg['loss']['transition_matrix']}")
    else:
        ortho_criterion = None
        lambda_ortho = 0.0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(args.epochs if args.epochs is not None else cfg["train"]["epochs"])
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    scheduler_name = cfg["train"].get("scheduler", "none")
    base_lr = float(cfg["train"]["lr"])
    min_lr = float(cfg["train"].get("min_lr", 0.0))
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 5.0))
    if args.debug:
        grad_accum_steps = 1
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    def lr_lambda(current_step: int):
        if scheduler_name == "none":
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if scheduler_name == "cosine":
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_lr_ratio = min_lr / max(base_lr, 1e-12)
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )
    logger.info(
        f"Gradient accumulation: {grad_accum_steps}, "
        f"effective_batch_size={batch_size * grad_accum_steps}, "
        f"grad_clip_norm={grad_clip_norm}"
    )
    logger.info(
        f"Scheduler: {scheduler_name}, warmup_epochs={warmup_epochs}, "
        f"total_steps={total_steps}, warmup_steps={warmup_steps}"
    )
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    best_word_accuracy = -1.0
    best_metric_value = -1.0
    epochs_since_improvement = 0
    history = []
    start_epoch = 1

    resume_path = Path(args.resume) if args.resume else None
    if args.auto_resume and resume_path is None:
        candidate = ckpt_dir / "last.pth"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)

        if "model" not in checkpoint:
            raise KeyError(f"Checkpoint does not contain key 'model': {resume_path}")

        if ema_model is not None and "train_model" not in checkpoint:
            logger.warning(
                "Resume checkpoint has no raw train_model weights. "
                "This is an older checkpoint format; if this was an interrupted "
                "DAB run, restarting that experiment from scratch is safer."
            )

        train_model_state = checkpoint.get("train_model", checkpoint["model"])
        model.load_state_dict(train_model_state, strict=True)

        if ema_model is not None:
            ema_state = checkpoint.get("ema_model") or checkpoint["model"]
            ema_model.load_state_dict(ema_state, strict=True)

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        else:
            logger.warning(f"Checkpoint has no optimizer state: {resume_path}")

        completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        completed_steps = completed_epoch * steps_per_epoch
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            scheduler.last_epoch = completed_steps
            lr_factor = lr_lambda(completed_steps)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_factor

        if "scaler" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler"])
            except Exception as exc:
                logger.warning(f"Failed to restore AMP scaler state: {exc!r}")

        best_word_accuracy = float(checkpoint.get("best_word_accuracy", best_word_accuracy))
        best_metric_name_from_ckpt = checkpoint.get("best_metric_name")
        best_metric_value = float(checkpoint.get("best_metric_value", best_metric_value))

        metrics_path = metrics_dir / "metrics.json"
        if metrics_path.exists():
            try:
                import json

                with metrics_path.open("r", encoding="utf-8") as f:
                    history = [
                        item
                        for item in json.load(f)
                        if int(item.get("epoch", 0)) <= completed_epoch
                    ]
            except Exception as exc:
                logger.warning(f"Failed to load existing history: {exc!r}")

        logger.info(
            f"Resumed from: {resume_path}, "
            f"completed_epoch={completed_epoch}, "
            f"start_epoch={start_epoch}, "
            f"lr={optimizer.param_groups[0]['lr']:.8f}, "
            f"best_word_accuracy={best_word_accuracy:.4f}, "
            f"best_metric={best_metric_name_from_ckpt or 'unknown'}:"
            f"{best_metric_value:.4f}"
        )

    val_interval = int(cfg["train"].get("val_interval", 50))
    rerank_val_interval = int(cfg["train"].get("rerank_val_interval", 0))
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 0))
    logger.info(
        f"Validation interval: every {val_interval} epoch(s), "
        f"enabled={validate_during_training}"
    )
    if validate_during_training:
        ctc_lexicon_reranker = build_ctc_lexicon_reranker(
            cfg=cfg,
            train_list=train_list,
            converter=converter,
            logger=logger,
        )
    else:
        ctc_lexicon_reranker = None
        rerank_val_interval = 0
    if args.debug and ctc_lexicon_reranker is not None:
        rerank_val_interval = val_interval
    best_metric_name = cfg["train"].get(
        "best_metric",
        "ctc_rerank_word_accuracy"
        if ctc_lexicon_reranker is not None
        else "word_accuracy",
    )
    logger.info(
        f"Rerank validation interval: "
        f"{rerank_val_interval if rerank_val_interval > 0 else 'disabled'}"
    )
    logger.info(f"Best checkpoint metric: {best_metric_name}")
    if early_stop_patience > 0:
        logger.info(
            f"Early stopping enabled. patience={early_stop_patience} validation check(s)"
        )

    if start_epoch > epochs:
        completed_epoch = start_epoch - 1
        best_checkpoint_path = ckpt_dir / "best.pth"

        if best_checkpoint_path.is_file():
            logger.info(
                f"Checkpoint epoch {completed_epoch} is already >= target epochs "
                f"{epochs}, and best.pth already exists. Nothing to train."
            )
            return

        if val_loader is None:
            raise RuntimeError(
                "Training is already complete but best.pth is missing, and "
                "validation is disabled. Enable validate_during_training to "
                "recover a validated best checkpoint."
            )

        logger.info(
            f"Checkpoint epoch {completed_epoch} is already >= target epochs "
            f"{epochs}, but best.pth is missing. Validating last.pth before "
            "saving it as the best available checkpoint."
        )
        recovery_lambda_align = apply_loss_warmup(
            weight=scheduled_weight(
                initial=lambda_align_initial,
                final=lambda_align_final,
                start_epoch=lambda_align_decay_start,
                end_epoch=lambda_align_decay_end,
                epoch=completed_epoch,
            ),
            epoch=completed_epoch,
            activate_epoch=lambda_align_activate_epoch,
            warmup_epochs=lambda_align_warmup_epochs,
        )
        recovery_lambda_ortho = lambda_ortho
        recovery_metrics = validate(
            model=ema_model if ema_model is not None else model,
            dataloader=val_loader,
            criterion=criterion,
            converter=converter,
            device=device,
            max_batches=args.max_val_batches,
            ortho_criterion=ortho_criterion,
            lambda_ortho=recovery_lambda_ortho,
            align_criterion=align_criterion,
            lambda_align=recovery_lambda_align,
            lexicon_corrector=lexicon_corrector,
            ctc_lexicon_reranker=(
                ctc_lexicon_reranker if rerank_val_interval > 0 else None
            ),
            channels_last=use_channels_last,
        )
        best_metric_value = float(
            recovery_metrics.get(best_metric_name, float("nan"))
        )
        if not math.isfinite(best_metric_value):
            raise RuntimeError(
                f"Best-checkpoint metric {best_metric_name!r} is not finite during "
                "completed-run checkpoint recovery."
            )
        best_word_accuracy = float(recovery_metrics["word_accuracy"])

        save_checkpoint(
            path=best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=completed_epoch,
            best_word_accuracy=best_word_accuracy,
            cfg=cfg,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            ema_model=ema_model,
            scheduler=scheduler,
            scaler=scaler,
        )
        save_checkpoint(
            path=ckpt_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            epoch=completed_epoch,
            best_word_accuracy=best_word_accuracy,
            cfg=cfg,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            ema_model=ema_model,
            scheduler=scheduler,
            scaler=scaler,
        )
        logger.info(
            f"Recovered best.pth from validated last.pth: "
            f"{best_metric_name}={best_metric_value:.4f}, "
            f"WA={best_word_accuracy:.4f}."
        )
        return

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        total_loss = 0.0
        total_ctc_loss = 0.0
        total_nrtr_loss = 0.0
        total_ortho_loss = 0.0
        total_weighted_ortho_loss = 0.0
        total_batches = 0
        current_lambda_nrtr = scheduled_weight(
            initial=lambda_nrtr_initial,
            final=lambda_nrtr_final,
            start_epoch=lambda_nrtr_decay_start,
            end_epoch=lambda_nrtr_decay_end,
            epoch=epoch,
        )
        current_lambda_nrtr = apply_loss_warmup(
            weight=current_lambda_nrtr,
            epoch=epoch,
            activate_epoch=lambda_nrtr_activate_epoch,
            warmup_epochs=lambda_nrtr_warmup_epochs,
        )
        current_lambda_align = scheduled_weight(
            initial=lambda_align_initial,
            final=lambda_align_final,
            start_epoch=lambda_align_decay_start,
            end_epoch=lambda_align_decay_end,
            epoch=epoch,
        )
        current_lambda_align = apply_loss_warmup(
            weight=current_lambda_align,
            epoch=epoch,
            activate_epoch=lambda_align_activate_epoch,
            warmup_epochs=lambda_align_warmup_epochs,
        )
        current_lambda_ortho = lambda_ortho

        for batch_idx, batch in enumerate(train_loader, start=1):
            if args.max_train_batches is not None and batch_idx > args.max_train_batches:
                break

            batch = move_batch_to_device(batch, device, channels_last=use_channels_last)

            if batch_idx == 1:
                optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                if nrtr_criterion is not None and current_lambda_nrtr > 0:
                    train_outputs = model.forward_train(
                        images=batch["images"],
                        targets=batch["targets"],
                        target_lengths=batch["target_lengths"],
                    )
                    logits = train_outputs["ctc_logits"]
                    nrtr_logits = train_outputs["nrtr_logits"]
                    nrtr_targets = train_outputs["nrtr_targets"]
                else:
                    logits = model(batch["images"])
                    nrtr_logits = None
                    nrtr_targets = None

            if epoch == 1 and batch_idx == 1:
                logger.info(
                    f"Logits shape: {tuple(logits.shape)}, "
                    f"image_shape={tuple(batch['images'].shape)}, "
                    f"target_len_min={batch['target_lengths'].min().item()}, "
                    f"target_len_max={batch['target_lengths'].max().item()}"
                )
                if nrtr_logits is not None:
                    logger.info(
                        f"NRTR logits shape: {tuple(nrtr_logits.shape)}, "
                        f"targets_shape={tuple(nrtr_targets.shape)}"
                    )

            logits_for_loss = logits.float()
            ctc_loss = criterion(
                logits=logits_for_loss,
                targets=batch["targets"],
                target_lengths=batch["target_lengths"],
            )
            if nrtr_logits is not None and nrtr_targets is not None:
                nrtr_loss = nrtr_criterion(
                    logits=nrtr_logits.float(),
                    targets=nrtr_targets,
                )
            else:
                nrtr_loss = logits_for_loss.new_tensor(0.0)
            if ortho_criterion is not None and current_lambda_ortho > 0:
                ortho_loss = ortho_criterion(logits_for_loss)
            else:
                ortho_loss = logits_for_loss.new_tensor(0.0)

            if align_criterion is not None and current_lambda_align > 0:
                align_loss = align_criterion(
                    logits=logits_for_loss,
                    targets=batch["targets"],
                    target_lengths=batch["target_lengths"],
                )
            else:
                align_loss = logits_for_loss.new_tensor(0.0)

            loss = (
                ctc_loss
                + current_lambda_nrtr * nrtr_loss
                + current_lambda_ortho * ortho_loss
                + current_lambda_align * align_loss
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch={epoch}, batch={batch_idx}. "
                    f"CTC={ctc_loss.item()}, NRTR={nrtr_loss.item()}, "
                    f"Ortho={ortho_loss.item()}, "
                    f"Align={align_loss.item()}"
                )
            loss_to_backward = loss / grad_accum_steps
            scaler.scale(loss_to_backward).backward()
            should_step = (
                batch_idx % grad_accum_steps == 0
                or batch_idx == len(train_loader)
                or (
                    args.max_train_batches is not None
                    and batch_idx == args.max_train_batches
                )
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()

                if (not use_amp) or scaler.get_scale() >= old_scale:
                    scheduler.step()
                    if ema_model is not None:
                        update_ema_model(ema_model, model, ema_decay)

                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            total_nrtr_loss += nrtr_loss.item()
            batch_ortho_loss = ortho_loss.item()
            total_ortho_loss += batch_ortho_loss
            total_weighted_ortho_loss += (
                current_lambda_ortho * batch_ortho_loss
            )
            total_batches += 1

            if batch_idx % int(cfg["train"].get("log_interval", 50)) == 0:
                logger.info(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"LR: {optimizer.param_groups[0]['lr']:.8f} "
                    f"Loss: {loss.item():.4f} "
                    f"CTC: {ctc_loss.item():.4f} "
                    f"NRTR: {nrtr_loss.item():.4f} "
                    f"OTP: {batch_ortho_loss:.4f} "
                    f"Align: {align_loss.item():.4f} "
                    f"lambda_nrtr={current_lambda_nrtr:.6f} "
                    f"lambda_ortho={current_lambda_ortho:.6f} "
                    f"lambda_align={current_lambda_align:.6f}"
                )

        train_loss = total_loss / max(total_batches, 1)
        train_ctc_loss = total_ctc_loss / max(total_batches, 1)
        train_nrtr_loss = total_nrtr_loss / max(total_batches, 1)
        train_ortho_loss = total_ortho_loss / max(total_batches, 1)
        train_weighted_ortho_loss = (
            total_weighted_ortho_loss / max(total_batches, 1)
        )

        # Validate only at the configured interval, plus the final/debug epoch.
        # Forcing epoch 1 would bypass the rerank schedule and best metric.
        should_validate = validate_during_training and (
            epoch % val_interval == 0
            or epoch == epochs
            or args.debug
        )

        if should_validate:
            validation_model = ema_model if ema_model is not None else model
            should_rerank_validate = (
                ctc_lexicon_reranker is not None
                and rerank_val_interval > 0
                and (
                    epoch % rerank_val_interval == 0
                    or epoch == epochs
                    or args.debug
                )
            )
            if should_rerank_validate:
                logger.info("CTC rerank validation enabled for this epoch.")
            val_metrics = validate(
                model=validation_model,
                dataloader=val_loader,
                criterion=criterion,
                converter=converter,
                device=device,
                max_batches=args.max_val_batches,
                ortho_criterion=ortho_criterion,
                lambda_ortho=current_lambda_ortho,
                align_criterion=align_criterion,
                lambda_align=current_lambda_align,
                lexicon_corrector=lexicon_corrector,
                ctc_lexicon_reranker=(
                    ctc_lexicon_reranker if should_rerank_validate else None
                ),
                channels_last=use_channels_last,
            )
        else:
            val_metrics = {
                "loss": float("nan"),
                "ctc_loss": float("nan"),
                "ortho_loss": float("nan"),
                "align_loss": float("nan"),
                "word_accuracy": float("nan"),
                "character_accuracy": float("nan"),
                "cer": float("nan"),
                "edit_distance": float("nan"),
                "exact_word_accuracy": float("nan"),
                "strict_character_accuracy": float("nan"),
                "strict_cer": float("nan"),
                "ctc_rerank_word_accuracy": float("nan"),
                "ctc_rerank_character_accuracy": float("nan"),
                "ctc_rerank_cer": float("nan"),
                "ctc_rerank_edit_distance": float("nan"),
                "ctc_rerank_enabled": False,
            }

        current = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_accum_steps": int(grad_accum_steps),
            "effective_batch_size": int(batch_size * grad_accum_steps),
            "lambda_nrtr": float(current_lambda_nrtr),
            "lambda_align": float(current_lambda_align),
            "lambda_ortho": float(current_lambda_ortho),
            "train_loss": float(train_loss),
            "train_ctc_loss": float(train_ctc_loss),
            "train_nrtr_loss": float(train_nrtr_loss),
            "train_ortho_loss": float(train_ortho_loss),
            "train_weighted_ortho_loss": float(train_weighted_ortho_loss),
            "val_enabled": bool(should_validate),
            "val_loss": float(val_metrics["loss"]),
            "val_ctc_loss": float(val_metrics.get("ctc_loss", val_metrics["loss"])),
            "val_ortho_loss": float(val_metrics.get("ortho_loss", 0.0)),
            "val_align_loss": float(val_metrics.get("align_loss", 0.0)),
            "val_word_accuracy": float(val_metrics["word_accuracy"]),
            "val_exact_word_accuracy": float(val_metrics.get("exact_word_accuracy", val_metrics["word_accuracy"])),
            "val_character_accuracy": float(val_metrics["character_accuracy"]),
            "val_cer": float(val_metrics["cer"]),
            "val_strict_character_accuracy": float(val_metrics.get("strict_character_accuracy", val_metrics["character_accuracy"])),
            "val_strict_cer": float(val_metrics.get("strict_cer", val_metrics["cer"])),
            "val_edit_distance": float(val_metrics["edit_distance"]),
            "val_ctc_rerank_word_accuracy": float(val_metrics.get("ctc_rerank_word_accuracy", float("nan"))),
            "val_ctc_rerank_character_accuracy": float(val_metrics.get("ctc_rerank_character_accuracy", float("nan"))),
            "val_ctc_rerank_cer": float(val_metrics.get("ctc_rerank_cer", float("nan"))),
            "val_ctc_rerank_edit_distance": float(val_metrics.get("ctc_rerank_edit_distance", float("nan"))),
            "val_ctc_rerank_enabled": bool(val_metrics.get("ctc_rerank_enabled", False)),
        }

        history.append(current)
        save_json(history, metrics_dir / "metrics.json")

        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"lr={optimizer.param_groups[0]['lr']:.8f} "
            f"lambda_nrtr={current_lambda_nrtr:.6f} "
            f"lambda_align={current_lambda_align:.6f} "
            f"lambda_ortho={current_lambda_ortho:.6f} "
            f"train_loss={train_loss:.4f} "
            f"train_ctc={train_ctc_loss:.4f} "
            f"train_nrtr={train_nrtr_loss:.4f} "
            f"train_otp={train_ortho_loss:.4f} "
            f"train_weighted_otp={train_weighted_ortho_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_ctc={val_metrics.get('ctc_loss', val_metrics['loss']):.4f} "
            f"val_ortho={val_metrics.get('ortho_loss', 0.0):.4f} "
            f"val_align={val_metrics.get('align_loss', 0.0):.4f} "
            f"WA={val_metrics['word_accuracy']:.4f} "
            f"CA={val_metrics['character_accuracy']:.4f} "
            f"CER={val_metrics['cer']:.4f} "
            f"ctc_WA={val_metrics.get('ctc_rerank_word_accuracy', float('nan')):.4f} "
            f"ctc_CER={val_metrics.get('ctc_rerank_cer', float('nan')):.4f}"
        )

        current_best_metric = float(val_metrics.get(best_metric_name, float("nan")))
        best_checkpoint_path = ckpt_dir / "best.pth"
        stop_training = False

        if should_validate:
            if not math.isfinite(current_best_metric):
                raise RuntimeError(
                    f"Best-checkpoint metric {best_metric_name!r} is not finite at "
                    f"epoch={epoch}. Ensure that the configured validation decoder "
                    "runs every validation epoch."
                )

            metric_improved = current_best_metric > best_metric_value
            best_checkpoint_missing = not best_checkpoint_path.is_file()

            if metric_improved or best_checkpoint_missing:
                if best_checkpoint_missing and not metric_improved:
                    logger.warning(
                        "best.pth is missing while resuming; the current validated "
                        "checkpoint becomes the best available checkpoint."
                    )

                best_metric_value = current_best_metric
                best_word_accuracy = val_metrics["word_accuracy"]
                epochs_since_improvement = 0

                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_word_accuracy=best_word_accuracy,
                    cfg=cfg,
                    best_metric_name=best_metric_name,
                    best_metric_value=best_metric_value,
                    ema_model=ema_model,
                    scheduler=scheduler,
                    scaler=scaler,
                )

                logger.info(
                    f"Saved best checkpoint. "
                    f"Best {best_metric_name}={best_metric_value:.4f}, "
                    f"WA={best_word_accuracy:.4f}, "
                    f"ctc_WA={val_metrics.get('ctc_rerank_word_accuracy', float('nan')):.4f}"
                )
            elif early_stop_patience > 0:
                epochs_since_improvement += 1
                logger.info(
                    f"No {best_metric_name} improvement for "
                    f"{epochs_since_improvement}/{early_stop_patience} "
                    "validation check(s)."
                )

                if epochs_since_improvement >= early_stop_patience:
                    stop_training = True

        # Save last.pth after updating the best state so auto-resume restores
        # checkpoint-selection metadata from the same validation epoch.
        save_checkpoint(
            path=ckpt_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_word_accuracy=best_word_accuracy,
            cfg=cfg,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            ema_model=ema_model,
            scheduler=scheduler,
            scaler=scaler,
        )

        if stop_training:
            logger.info(
                f"Early stopping triggered at epoch={epoch}. "
                f"Best {best_metric_name}={best_metric_value:.4f}"
            )
            break

        if args.debug:
            logger.info("Debug run finished after one epoch.")
            break

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
