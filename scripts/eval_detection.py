import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from manchu_ocr.data.collate.det_collate import DetCollate
from manchu_ocr.data.datasets.detection_dataset import DetectionDataset
from manchu_ocr.data.label_generators.db_label_generator import DBLabelGenerator
from manchu_ocr.data.transforms.det_transforms import build_det_transform
from manchu_ocr.metrics.detection_metrics import DetectionMetric
from manchu_ocr.models.detection.builder import build_detection_model
from manchu_ocr.models.detection.postprocess.db_postprocess import DBPostProcessor
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


STRICT_IOU_THRESHOLD = 0.75


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().float()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    # RGB -> BGR
    return image[:, :, ::-1].copy()


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    encoded.tofile(str(path))


def draw_predictions(
    image: np.ndarray,
    pred_boxes: List,
    gt_polygons: List[Dict],
) -> np.ndarray:
    canvas = image.copy()

    # GT: green
    for poly in gt_polygons:
        pts = np.asarray(poly["points"], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Prediction: red
    for box in pred_boxes:
        pts = np.asarray(box, dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    return canvas


def build_dataloader(
    manifest_path: str,
    target_height: int,
    target_width: int,
    shrink_ratio: float,
    batch_size: int,
    num_workers: int,
    label_generator_cfg: dict,
    degrade: bool = False,
    degradation_cfg: dict | None = None,
) -> DataLoader:
    transform = build_det_transform(
        target_height=target_height,
        target_width=target_width,
        keep_aspect_ratio=True,
        augment=degrade,
        augmentation_cfg=degradation_cfg,
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
        shuffle=False,
        num_workers=num_workers,
        collate_fn=DetCollate(),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataloader


def build_model(cfg: Dict):
    return build_detection_model(cfg)


def build_device(cfg: Dict) -> torch.device:
    runtime_cfg = cfg.get("runtime", {})
    requested_device = runtime_cfg.get("device", "auto")
    allow_cpu = bool(runtime_cfg.get("allow_cpu", True))

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requires CUDA, but CUDA is unavailable.")
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError("Evaluation is running on CPU, but allow_cpu=false.")

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

    neck = getattr(model, "neck", None)
    direction_module_name = getattr(neck, "direction_module_name", None)
    if direction_module_name is not None:
        logger.info(f"Direction module: {direction_module_name}")

    logger.info(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model) / 1_000_000:.3f}M"
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    postprocessor: DBPostProcessor,
    metric: DetectionMetric,
    device: torch.device,
    output_vis_dir: Path | None = None,
    max_batches: int | None = None,
) -> tuple[Dict, List[Dict]]:
    model.eval()

    records = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device, non_blocking=True)

        preds = model(images)
        post_results = postprocessor(preds)

        batch_size = images.shape[0]

        for i in range(batch_size):
            pred_boxes = post_results[i]["boxes"]
            pred_scores = post_results[i]["scores"]
            gt_polygons = batch["polygons"][i]

            image_stat = metric.update(
                pred_boxes=pred_boxes,
                gt_polygons=gt_polygons,
            )

            image_path = batch["image_paths"][i]

            record = {
                "image_path": image_path,
                "num_gt": image_stat["num_gt"],
                "num_pred": image_stat["num_pred"],
                "tp": image_stat["tp"],
                "fp": image_stat["fp"],
                "fn": image_stat["fn"],
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
            }

            records.append(record)

            if output_vis_dir is not None:
                image = denormalize_image(batch["images"][i])
                vis = draw_predictions(
                    image=image,
                    pred_boxes=pred_boxes,
                    gt_polygons=gt_polygons,
                )

                out_name = f"{len(records):04d}_{Path(image_path).stem}.jpg"
                imwrite_unicode(output_vis_dir / out_name, vis)

    metrics = metric.compute()

    return metrics, records


def main() -> None:
    run_timer = ExperimentTimer()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--binary-thresh", type=float, default=None)
    parser.add_argument("--box-thresh", type=float, default=None)
    parser.add_argument("--unclip-ratio", type=float, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix for output files, e.g. _iou70.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths_cfg = load_yaml(cfg["experiment"]["paths_config"])

    set_seed(cfg["train"].get("seed", 42))

    exp_name = cfg["experiment"]["name"]

    logger = setup_logger(f"eval_detection_{exp_name}_{args.split}")

    device = build_device(cfg)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    det_cfg = paths_cfg["detection_data"]

    if args.split == "train":
        manifest_path = det_cfg["train_list"]
    elif args.split == "val":
        manifest_path = det_cfg["val_list"]
    else:
        manifest_path = det_cfg["test_list"]

    target_height = int(cfg["data"]["target_height"])
    target_width = int(cfg["data"]["target_width"])
    shrink_ratio = float(cfg["data"].get("shrink_ratio", 0.4))
    label_generator_cfg = cfg["data"].get("label_generator", {})
    dataloader = build_dataloader(
        manifest_path=manifest_path,
        target_height=target_height,
        target_width=target_width,
        shrink_ratio=shrink_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_generator_cfg=label_generator_cfg,
        degrade=False,
        degradation_cfg=None,
    )

    model = build_model(cfg).to(device)
    log_model_summary(logger, model, cfg)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"], strict=True)

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Loaded checkpoint epoch={checkpoint.get('epoch', 'unknown')}")
    logger.info("Evaluation protocol: original images, no evaluation degradation.")

    eval_cfg = cfg.get("eval", {})
    binary_thresh = float(args.binary_thresh if args.binary_thresh is not None else eval_cfg.get("binary_thresh", 0.3))
    box_thresh = float(args.box_thresh if args.box_thresh is not None else eval_cfg.get("box_thresh", 0.5))
    unclip_ratio = float(args.unclip_ratio if args.unclip_ratio is not None else eval_cfg.get("unclip_ratio", 1.5))
    iou_thresh = STRICT_IOU_THRESHOLD
    min_size = int(eval_cfg.get("min_size", 3))

    postprocessor = DBPostProcessor(
        binary_thresh=binary_thresh,
        box_thresh=box_thresh,
        unclip_ratio=unclip_ratio,
        min_size=min_size,
    )

    metric = DetectionMetric(
        iou_threshold=iou_thresh,
    )

    output_root = Path(paths_cfg["outputs"]["root"])
    metrics_dir = output_root / "metrics" / "detection" / exp_name
    pred_dir = output_root / "predictions" / "detection" / exp_name
    vis_dir = output_root / "visualizations" / "detection" / exp_name / args.split

    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    output_vis_dir = vis_dir if args.save_vis else None

    metrics, records = evaluate(
        model=model,
        dataloader=dataloader,
        postprocessor=postprocessor,
        metric=metric,
        device=device,
        output_vis_dir=output_vis_dir,
        max_batches=args.max_batches,
    )

    metrics["num_samples"] = int(len(records))

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timing = run_timer.finish()
    num_batches = len(dataloader)
    if args.max_batches is not None:
        num_batches = min(num_batches, max(0, int(args.max_batches)))

    runtime_details = collect_torch_runtime(device)
    runtime_details.update(
        {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "num_samples": int(len(records)),
            "num_batches": int(num_batches),
            "trainable_parameters": int(count_trainable_parameters(model)),
            "direction_module": getattr(
                getattr(model, "neck", None),
                "direction_module_name",
                None,
            ),
            "samples_per_second": round(
                float(len(records)) / max(timing.elapsed_seconds, 1e-12),
                6,
            ),
        }
    )

    model_architecture = str(
        cfg.get("model", {}).get("name", model.__class__.__name__)
    )
    result = build_experiment_result(
        task="detection",
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
            "direction_module": getattr(
                getattr(model, "neck", None),
                "direction_module_name",
                None,
            ),
            "legacy_split": args.split,
            "checkpoint": str(checkpoint_path).replace("\\", "/"),
            "checkpoint_epoch": checkpoint.get("epoch", None),
            "postprocess": {
                "binary_thresh": binary_thresh,
                "box_thresh": box_thresh,
                "unclip_ratio": unclip_ratio,
                "iou_thresh": iou_thresh,
                "min_size": min_size,
            },
            "evaluation_protocol": {
                "strict": True,
                "degradation_enabled": False,
                "iou_threshold": STRICT_IOU_THRESHOLD,
            },
        },
    )

    suffix = args.output_suffix.strip()

    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    save_json(result, metrics_dir / f"eval_{args.split}{suffix}.json")

    if args.save_predictions:
        save_json(records, pred_dir / f"{args.split}{suffix}_predictions.json")

    logger.info(
        f"Evaluation finished. "
        f"P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, "
        f"F={metrics['fmeasure']:.4f}, "
        f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, "
        f"GT={metrics['num_gt']}, Pred={metrics['num_pred']}"
    )
    logger.info(format_experiment_result(result))

    logger.info(f"Saved metrics to: {metrics_dir / f'eval_{args.split}{suffix}.json'}")

    if args.save_vis:
        logger.info(f"Saved visualizations to: {vis_dir}")


if __name__ == "__main__":
    main()
