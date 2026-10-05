from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from manchu_ocr.data.label_converters.ctc_label_converter import CTCLabelConverter
from manchu_ocr.data.transforms.rec_transforms import build_rec_transform
from manchu_ocr.models.recognition.builder import build_recognition_model
from manchu_ocr.models.recognition.decoders.lexicon_corrector import (
    CTCLexiconReranker,
    LexiconCorrector,
    load_lexicon_from_manifest,
)
from manchu_ocr.utils.config import load_yaml


class RecognitionPipeline:
    """Run CTC recognition on one word crop."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        charset_path: str | Path | None = None,
        device: str | torch.device = "cuda",
    ):
        self.cfg = load_yaml(config_path)
        paths_cfg = load_yaml(self.cfg["experiment"]["paths_config"])
        charset_path = charset_path or paths_cfg["recognition_data"]["charset"]

        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.converter = CTCLabelConverter(charset_path)
        decode_cfg = self.cfg.get("decode", {})
        train_list = paths_cfg["recognition_data"]["train_list"]
        self.transform = build_rec_transform(
            image_height=int(self.cfg["data"]["image_height"]),
            image_width=int(self.cfg["data"]["image_width"]),
            keep_aspect_ratio=True,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_cfg = self.cfg

        checkpoint_cfg = checkpoint.get("config")
        if isinstance(checkpoint_cfg, dict) and "model" in checkpoint_cfg:
            # The YAML may have been tuned after training. Build the network from
            # the checkpoint config so full-page inference remains compatible.
            model_cfg = self.cfg.copy()
            model_cfg["model"] = checkpoint_cfg["model"]

        self.model = build_recognition_model(model_cfg, num_classes=self.converter.num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()

        self.lexicon_corrector = None
        self.ctc_lexicon_reranker = None

        if bool(decode_cfg.get("use_lexicon", False)):
            lexicon_counts = load_lexicon_from_manifest(train_list)

            if bool(decode_cfg.get("use_ctc_lexicon_rerank", False)):
                self.ctc_lexicon_reranker = CTCLexiconReranker(
                    lexicon_counts=lexicon_counts,
                    char_to_idx=self.converter.char_to_idx,
                    blank_idx=self.converter.blank_idx,
                    max_edit_distance=int(decode_cfg.get("ctc_rerank_max_edit_distance", 3)),
                    length_delta=int(decode_cfg.get("ctc_rerank_length_delta", 3)),
                    min_word_length=int(decode_cfg.get("ctc_rerank_min_word_length", 2)),
                    max_candidates=int(decode_cfg.get("ctc_rerank_max_candidates", 80)),
                    prior_weight=float(decode_cfg.get("ctc_rerank_prior_weight", 0.04)),
                )
            else:
                self.lexicon_corrector = LexiconCorrector(
                    lexicon_counts=lexicon_counts,
                    max_edit_distance=int(decode_cfg.get("max_edit_distance", 3)),
                    length_delta=int(decode_cfg.get("length_delta", 3)),
                    min_word_length=int(decode_cfg.get("min_word_length", 2)),
                )

    @torch.no_grad()
    def __call__(self, image: str | Path | Image.Image) -> Dict:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        logits = logits.float()
        raw_text = self.converter.decode_logits(logits.detach().cpu())[0]

        if self.ctc_lexicon_reranker is not None:
            text = self.ctc_lexicon_reranker.rerank_batch(logits, [raw_text])[0]
        elif self.lexicon_corrector is not None:
            text = self.lexicon_corrector.correct(raw_text)
        else:
            text = raw_text

        confidence = float(torch.softmax(logits.float(), dim=-1).max(dim=-1).values.mean().item())
        return {"text": text, "raw_text": raw_text, "confidence": confidence}
