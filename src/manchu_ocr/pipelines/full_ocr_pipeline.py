from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

from manchu_ocr.pipelines.crop_rectify import crop_polygon_bbox
from manchu_ocr.pipelines.detect import DetectionPipeline
from manchu_ocr.pipelines.recognize import RecognitionPipeline


class FullOCRPipeline:
    """Detection followed by word-crop recognition."""

    def __init__(
        self,
        detection_config: str | Path,
        detection_checkpoint: str | Path,
        recognition_config: str | Path,
        recognition_checkpoint: str | Path,
        device: str = "cuda",
        crop_padding: int = 4,
        crop_padding_ratio: float = 0.04,
    ):
        self.detector = DetectionPipeline(detection_config, detection_checkpoint, device=device)
        self.recognizer = RecognitionPipeline(recognition_config, recognition_checkpoint, device=device)
        self.crop_padding = crop_padding
        self.crop_padding_ratio = crop_padding_ratio

    @staticmethod
    def _bbox(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def sort_reading_order(cls, boxes: List, scores: List[float]) -> tuple[List, List[float]]:
        """
        Sort detected word boxes into a deterministic reading order.

        Current Manchu page convention in this project is column-major:
        left-to-right by column, then top-to-bottom inside each column.
        """
        paired = list(zip(boxes, scores))
        paired = sorted(paired, key=lambda item: (cls._bbox(item[0])[0], cls._bbox(item[0])[1]))

        if not paired:
            return [], []

        sorted_boxes, sorted_scores = zip(*paired)
        return list(sorted_boxes), [float(score) for score in sorted_scores]

    @classmethod
    def group_columns(cls, items: List[Dict]) -> List[Dict]:
        """
        Group recognized word items into coarse page columns.

        The output is meant for document-level reading/export, not for metrics.
        Manchu pages in this project are represented as column-major order:
        columns from left to right, words from top to bottom inside a column.
        """
        if not items:
            return []

        def center_x(item: Dict) -> float:
            x1, _, x2, _ = cls._bbox(item["box"])
            return (x1 + x2) / 2.0

        def width(item: Dict) -> float:
            x1, _, x2, _ = cls._bbox(item["box"])
            return max(1.0, x2 - x1)

        widths = sorted(width(item) for item in items)
        median_width = widths[len(widths) // 2]
        max_same_column_gap = max(10.0, median_width * 1.6)

        columns: List[List[Dict]] = []

        for item in sorted(items, key=center_x):
            cx = center_x(item)

            if not columns:
                columns.append([item])
                continue

            last_column = columns[-1]
            last_cx = sum(center_x(it) for it in last_column) / len(last_column)

            if abs(cx - last_cx) <= max_same_column_gap:
                last_column.append(item)
            else:
                columns.append([item])

        output = []

        for col_idx, column_items in enumerate(columns):
            column_items = sorted(column_items, key=lambda item: cls._bbox(item["box"])[1])
            text = " ".join(str(item.get("text", "")).strip() for item in column_items if str(item.get("text", "")).strip())

            output.append(
                {
                    "column_index": col_idx,
                    "text": text,
                    "items": column_items,
                }
            )

        return output

    @staticmethod
    def build_page_text(columns: List[Dict]) -> str:
        """Represent the page as one text line per Manchu column."""
        return "\n".join(column["text"] for column in columns if column.get("text"))

    def __call__(self, image_path: str | Path) -> Dict:
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        det = self.detector(image)
        boxes, scores = self.sort_reading_order(det["boxes"], det["scores"])
        records: List[Dict] = []

        for idx, (box, score) in enumerate(zip(boxes, scores)):
            crop = crop_polygon_bbox(
                image,
                box,
                padding=self.crop_padding,
                padding_ratio=self.crop_padding_ratio,
            )
            rec = self.recognizer(crop)
            records.append(
                {
                    "index": idx,
                    "box": box,
                    "det_score": float(score),
                    "text": rec["text"],
                    "raw_text": rec.get("raw_text", rec["text"]),
                    "rec_confidence": rec["confidence"],
                }
            )

        columns = self.group_columns(records)
        page_text = self.build_page_text(columns)

        return {
            "image_path": str(image_path).replace("\\", "/"),
            "items": records,
            "columns": columns,
            "page_text": page_text,
            "flat_text": " ".join(item["text"] for item in records if item.get("text")),
        }
