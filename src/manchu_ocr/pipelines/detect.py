from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from manchu_ocr.data.transforms.det_transforms import build_det_transform
from manchu_ocr.models.detection.builder import build_detection_model
from manchu_ocr.models.detection.postprocess.db_postprocess import DBPostProcessor
from manchu_ocr.models.detection.postprocess.polygon_restore import restore_polygons
from manchu_ocr.utils.config import load_yaml


class DetectionPipeline:
    """Run DBNet-style detection on one page image."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
        binary_thresh: float | None = None,
        box_thresh: float | None = None,
        unclip_ratio: float | None = None,
    ):
        self.cfg = load_yaml(config_path)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")

        self.transform = build_det_transform(
            target_height=int(self.cfg["data"]["target_height"]),
            target_width=int(self.cfg["data"]["target_width"]),
            keep_aspect_ratio=True,
        )

        eval_cfg = self.cfg.get("eval", {})
        self.postprocessor = DBPostProcessor(
            binary_thresh=float(binary_thresh if binary_thresh is not None else eval_cfg.get("binary_thresh", 0.3)),
            box_thresh=float(box_thresh if box_thresh is not None else eval_cfg.get("box_thresh", 0.5)),
            unclip_ratio=float(unclip_ratio if unclip_ratio is not None else eval_cfg.get("unclip_ratio", 1.5)),
            min_size=int(eval_cfg.get("min_size", 3)),
        )

        self.model = build_detection_model(self.cfg).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image: str | Path | Image.Image) -> Dict:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor, _, meta = self.transform(image, [])
        preds = self.model(tensor.unsqueeze(0).to(self.device))
        result = self.postprocessor(preds)[0]
        boxes = restore_polygons(result["boxes"], meta)

        return {
            "boxes": boxes,
            "scores": result["scores"],
            "meta": meta,
        }
