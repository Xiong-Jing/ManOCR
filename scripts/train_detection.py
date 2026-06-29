import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from manchu_ocr.data.collate.det_collate import DetCollate
from manchu_ocr.data.datasets.detection_dataset import DetectionDataset
from manchu_ocr.data.label_generators.db_label_generator import DBLabelGenerator
from manchu_ocr.data.transforms.det_transforms import build_det_transform
from manchu_ocr.losses.db_loss import DBLoss
from manchu_ocr.models.detection.builder import build_detection_model
from manchu_ocr.metrics.detection_metrics import DetectionMetric
from manchu_ocr.models.detection.postprocess.db_postprocess import DBPostProcessor
from manchu_ocr.utils.config import load_yaml
from manchu_ocr.utils.file_io import save_json
from manchu_ocr.utils.logger import setup_logger
from manchu_ocr.utils.seed import set_seed


def build_dataloader(
    manifest_path: str,
    target_height: int,
    target_width: int,
    shrink_ratio: float,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    label_generator_cfg: dict,
    augment: bool = False,
    augmentation_cfg: Optional[dict] = None,
) -> DataLoader:
    transform = build_det_transform(
        target_height=target_height,
        target_width=target_width,
        keep_aspect_ratio=True,
        augment=augment,
        augmentation_cfg=augmentation_cfg,
    )

    label_generator = DBLabelGenerator(
        shrink_ratio=shrink_ratio,
        thresh_min=0.3,
        thresh_max=0.7,
        min_text_size=3,
        use_asymmetric_shrink=bool(label_generator_cfg.get("use_asymmetric_shrink", False)),
        shrink_ratio_x=float(label_generator_cfg.get("shrink_ratio_x", 0.65)),
        shrink_ratio_y=float(label_generator_cfg.get("shrink_ratio_y", 0.90)),
        as_auxiliary=bool(label_generator_cfg.get("as_auxiliary", True)),
    )

    dataset = DetectionDataset(
        manifest_path=manifest_path,
        transform=transform,
        label_generator=label_generator,
        check_exists=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=DetCollate(),
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )

    return dataloader


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    batch["images"] = batch["images"].to(device, non_blocking=True)

    for key in [
        "prob_map",
        "as_prob_map",
        "thresh_map",
        "thresh_mask",
        "training_mask",
    ]:
        if key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

    return batch


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: DBLoss,
    device: torch.device,
    postprocessor: Optional[DBPostProcessor] = None,
    iou_threshold: float = 0.5,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_prob_loss = 0.0
    total_binary_loss = 0.0
    total_thresh_loss = 0.0
    total_as_loss = 0.0
    total_batches = 0
    metric = DetectionMetric(iou_threshold=iou_threshold)
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        with autocast("cuda", enabled=False):
            preds = model(batch["images"])
        losses = criterion(preds, batch)
        total_loss += losses["loss"].item()
        total_prob_loss += losses["prob_loss"].item()
        total_binary_loss += losses["binary_loss"].item()
        total_thresh_loss += losses["thresh_loss"].item()
        total_as_loss += losses.get("as_loss", losses["loss"] * 0.0).item()
        total_batches += 1
        if postprocessor is not None:
            post_results = postprocessor(preds)
            for i in range(len(post_results)):
                metric.update(
                    pred_boxes=post_results[i]["boxes"],
                    gt_polygons=batch["polygons"][i],
                )
    metric_values = metric.compute()
    return {
        "loss": total_loss / max(total_batches, 1),
        "prob_loss": total_prob_loss / max(total_batches, 1),
        "binary_loss": total_binary_loss / max(total_batches, 1),
        "thresh_loss": total_thresh_loss / max(total_batches, 1),
        "as_loss": total_as_loss / max(total_batches, 1),
        "precision": float(metric_values["precision"]),
        "recall": float(metric_values["recall"]),
        "fmeasure": float(metric_values["fmeasure"]),
        "tp": int(metric_values["tp"]),
        "fp": int(metric_values["fp"]),
        "fn": int(metric_values["fn"]),
        "num_gt": int(metric_values["num_gt"]),
        "num_pred": int(metric_values["num_pred"]),
    }

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    cfg: Dict,
    best_val_fmeasure: float = -1.0,
    scheduler: Optional[object] = None,
    scaler: Optional[GradScaler] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_fmeasure": best_val_fmeasure,
        "config": cfg,
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    torch.save(state, path)


def build_device(cfg: Dict) -> torch.device:
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
            "Training is running on CPU, but allow_cpu=false."
        )

    return device


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def log_model_summary(logger, model: torch.nn.Module, cfg: Dict) -> None:
    model_cfg = cfg.get("model", {})
    logger.info(f"Model config name: {model_cfg.get('name', 'unknown')}")
    logger.info(f"Model class: {model.__class__.__name__}")

    for attr in ["backbone", "neck", "head"]:
        module = getattr(model, attr, None)
        if module is not None:
            logger.info(f"Model {attr}: {module.__class__.__name__}")

    logger.info(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model) / 1_000_000:.3f}M"
    )
    logger.info(
        "Detection training protocol: shared DB-compatible supervision maps. "
        "Component losses are saved in metrics.json for debugging."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detection/dbnetpp_baseline.yaml",
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
    log_dir = output_root / "logs" / "detection" / exp_name
    ckpt_dir = output_root / "checkpoints" / "detection" / exp_name
    metrics_dir = output_root / "metrics" / "detection" / exp_name

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name=f"train_detection_{exp_name}",
        log_file=log_dir / "train.log",
    )

    shutil.copyfile(args.config, log_dir / "config.yaml")

    det_cfg = paths_cfg["detection_data"]

    train_list = det_cfg["train_list"]
    val_list = det_cfg["val_list"]

    device = build_device(cfg)

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Train list: {train_list}")
    logger.info(f"Val list: {val_list}")

    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    target_height = int(cfg["data"]["target_height"])
    target_width = int(cfg["data"]["target_width"])
    shrink_ratio = float(cfg["data"].get("shrink_ratio", 0.4))
    label_generator_cfg = cfg["data"].get("label_generator", {})
    augmentation_cfg = cfg["data"].get("augmentation", {})
    use_train_augmentation = bool(augmentation_cfg.get("enabled", False))
    eval_degradation_cfg = cfg.get("eval_degradation", augmentation_cfg)
    use_eval_degradation = bool(eval_degradation_cfg.get("enabled", False))

    if args.batch_size is not None:
        batch_size = int(args.batch_size)

    if args.num_workers is not None:
        num_workers = int(args.num_workers)

    if args.debug:
        batch_size = 1
        if args.max_train_batches is None:
            args.max_train_batches = 5
        if args.max_val_batches is None:
            args.max_val_batches = 2
        logger.info("Debug mode enabled.")

    train_loader = build_dataloader(
        manifest_path=train_list,
        target_height=target_height,
        target_width=target_width,
        shrink_ratio=shrink_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        label_generator_cfg=label_generator_cfg,
        augment=use_train_augmentation,
        augmentation_cfg=augmentation_cfg,
    )

    val_loader = build_dataloader(
        manifest_path=val_list,
        target_height=target_height,
        target_width=target_width,
        shrink_ratio=shrink_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        label_generator_cfg=label_generator_cfg,
        augment=use_eval_degradation,
        augmentation_cfg=eval_degradation_cfg if use_eval_degradation else None,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(
        f"Train augmentation enabled: {use_train_augmentation}, "
        f"cfg={augmentation_cfg}"
    )
    logger.info(
        f"Validation degradation enabled: {use_eval_degradation}, "
        f"cfg={eval_degradation_cfg if use_eval_degradation else {}}"
    )

    model = build_detection_model(cfg).to(device)
    log_model_summary(logger, model, cfg)

    criterion = DBLoss(
        alpha=float(cfg["loss"].get("alpha", 1.0)),
        beta=float(cfg["loss"].get("beta", 10.0)),
        gamma_as=float(cfg["loss"].get("gamma_as", 0.0)),
        negative_ratio=float(cfg["loss"].get("negative_ratio", 3.0)),
        eps=float(cfg["loss"].get("eps", 1e-6)),
    )

    eval_cfg = cfg.get("eval", {})
    postprocessor = DBPostProcessor(
        binary_thresh=float(eval_cfg.get("binary_thresh", 0.3)),
        box_thresh=float(eval_cfg.get("box_thresh", 0.5)),
        unclip_ratio=float(eval_cfg.get("unclip_ratio", 1.5)),
        min_size=int(eval_cfg.get("min_size", 3)),
    )
    val_iou_threshold = float(eval_cfg.get("iou_thresh", 0.5))
    logger.info(
        f"Eval postprocess: binary_thresh={postprocessor.binary_thresh}, "
        f"box_thresh={postprocessor.box_thresh}, "
        f"unclip_ratio={postprocessor.unclip_ratio}, "
        f"iou_thresh={val_iou_threshold}"
    )

    epochs = int(args.epochs if args.epochs is not None else cfg["train"]["epochs"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    scheduler_name = cfg["train"].get("scheduler", "none")
    base_lr = float(cfg["train"]["lr"])
    min_lr = float(cfg["train"].get("min_lr", 0.0))
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
            progress = min(max(progress, 0.0), 1.0)
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
        f"effective_batch_size={batch_size * grad_accum_steps}"
    )
    logger.info(
        f"Scheduler: {scheduler_name}, "
        f"base_lr={base_lr}, min_lr={min_lr}, "
        f"warmup_epochs={warmup_epochs}, "
        f"total_steps={total_steps}, warmup_steps={warmup_steps}"
    )

    best_val_loss = float("inf")
    best_val_fmeasure = -1.0
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

        model.load_state_dict(checkpoint["model"], strict=True)

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

        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_val_fmeasure = float(checkpoint.get("best_val_fmeasure", best_val_fmeasure))

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
                    if history:
                        history_best_fmeasure = max(
                            float(item.get("val_fmeasure", -1.0))
                            for item in history
                        )
                        best_val_fmeasure = max(best_val_fmeasure, history_best_fmeasure)
            except Exception as exc:
                logger.warning(f"Failed to load existing history: {exc!r}")

        logger.info(
            f"Resumed from: {resume_path}, "
            f"completed_epoch={completed_epoch}, "
            f"start_epoch={start_epoch}, "
            f"lr={optimizer.param_groups[0]['lr']:.8f}, "
            f"best_val_loss={best_val_loss:.4f}, "
            f"best_val_fmeasure={best_val_fmeasure:.4f}"
        )

    if start_epoch > epochs:
        logger.info(
            f"Checkpoint epoch {start_epoch - 1} is already >= target epochs {epochs}. "
            "Nothing to train."
        )
        return

    val_interval = int(cfg["train"].get("val_interval", 1))

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        total_loss = 0.0
        total_prob_loss = 0.0
        total_binary_loss = 0.0
        total_thresh_loss = 0.0
        total_as_loss = 0.0
        total_batches = 0

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader, start=1):
            if args.max_train_batches is not None and batch_idx > args.max_train_batches:
                break
            batch = move_batch_to_device(batch, device)
            with autocast("cuda", enabled=use_amp):
                preds = model(batch["images"])
            if epoch == 1 and batch_idx == 1:
                logger.info(
                    f"Prediction map shape: {tuple(preds['prob_map'].shape)}"
                )
                logger.info(
                    f"Target map shape: {tuple(batch['prob_map'].shape)}"
                )
            # DBLoss contains BCE, so compute it outside autocast.
            losses = criterion(preds, batch)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite detection loss detected at epoch={epoch}, batch={batch_idx}. "
                    f"loss={losses['loss'].item()}, "
                    f"prob={losses['prob_loss'].item()}, "
                    f"binary={losses['binary_loss'].item()}, "
                    f"thresh={losses['thresh_loss'].item()}"
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()

                if (not use_amp) or scaler.get_scale() >= old_scale:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
            total_loss += losses["loss"].item()
            total_prob_loss += losses["prob_loss"].item()
            total_binary_loss += losses["binary_loss"].item()
            total_thresh_loss += losses["thresh_loss"].item()
            total_as_loss += losses.get("as_loss", losses["loss"] * 0.0).item()
            total_batches += 1

            if batch_idx % int(cfg["train"].get("log_interval", 20)) == 0:
                logger.info(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"LR: {optimizer.param_groups[0]['lr']:.8f} "
                    f"loss={losses['loss'].item():.4f}"
                )

        train_loss = total_loss / max(total_batches, 1)
        train_prob_loss = total_prob_loss / max(total_batches, 1)
        train_binary_loss = total_binary_loss / max(total_batches, 1)
        train_thresh_loss = total_thresh_loss / max(total_batches, 1)
        train_as_loss = total_as_loss / max(total_batches, 1)

        should_validate = (epoch % val_interval == 0) or (epoch == epochs)

        if should_validate:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                postprocessor=postprocessor,
                iou_threshold=val_iou_threshold,
                max_batches=args.max_val_batches,
            )
        else:
            val_metrics = {
                "loss": float("nan"),
                "prob_loss": float("nan"),
                "binary_loss": float("nan"),
                "thresh_loss": float("nan"),
                "as_loss": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "fmeasure": float("nan"),
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "num_gt": 0,
                "num_pred": 0,
            }

        current = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_accum_steps": int(grad_accum_steps),
            "effective_batch_size": int(batch_size * grad_accum_steps),
            "train_loss": float(train_loss),
            "train_prob_loss": float(train_prob_loss),
            "train_binary_loss": float(train_binary_loss),
            "train_thresh_loss": float(train_thresh_loss),
            "train_as_loss": float(train_as_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_prob_loss": float(val_metrics["prob_loss"]),
            "val_binary_loss": float(val_metrics["binary_loss"]),
            "val_thresh_loss": float(val_metrics["thresh_loss"]),
            "val_as_loss": float(val_metrics.get("as_loss", 0.0)),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_fmeasure": float(val_metrics["fmeasure"]),
            "val_tp": int(val_metrics["tp"]),
            "val_fp": int(val_metrics["fp"]),
            "val_fn": int(val_metrics["fn"]),
            "val_num_gt": int(val_metrics["num_gt"]),
            "val_num_pred": int(val_metrics["num_pred"]),
        }

        history.append(current)
        save_json(history, metrics_dir / "metrics.json")

        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"lr={optimizer.param_groups[0]['lr']:.8f} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"P={val_metrics['precision']:.4f} "
            f"R={val_metrics['recall']:.4f} "
            f"F={val_metrics['fmeasure']:.4f}"
        )

        save_checkpoint(
            path=ckpt_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            cfg=cfg,
            best_val_fmeasure=best_val_fmeasure,
            scheduler=scheduler,
            scaler=scaler,
        )

        if should_validate and val_metrics["fmeasure"] > best_val_fmeasure:
            best_val_fmeasure = val_metrics["fmeasure"]
            best_val_loss = val_metrics["loss"]

            save_checkpoint(
                path=ckpt_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                cfg=cfg,
                best_val_fmeasure=best_val_fmeasure,
                scheduler=scheduler,
                scaler=scaler,
            )

            logger.info(
                f"Saved best checkpoint. "
                f"Best F={best_val_fmeasure:.4f}, "
                f"val_loss={best_val_loss:.4f}"
            )

        if args.debug:
            logger.info("Debug run finished after one epoch.")
            break

    logger.info("Detection training finished.")


if __name__ == "__main__":
    main()









