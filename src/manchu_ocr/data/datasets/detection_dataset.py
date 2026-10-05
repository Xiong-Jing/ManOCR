import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    """
    Dataset for Manchu word detection.

    Manifest format:
        image_path<TAB>clean_json_path

    Clean JSON format:
        {
          "image_path": "...",
          "image_width": ...,
          "image_height": ...,
          "polygons": [
            {
              "label": "manchu",
              "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
              "bbox": [x1,y1,x2,y2]
            }
          ]
        }
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Optional[Callable] = None,
        label_generator: Optional[Callable] = None,
        check_exists: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.label_generator = label_generator

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.samples = self._load_manifest(self.manifest_path)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in manifest: {self.manifest_path}")

        if check_exists:
            self._check_files()

    @staticmethod
    def _load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
        samples = []

        with manifest_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.rstrip("\n")

                if line.strip() == "":
                    continue

                parts = line.split("\t")

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid line format in {manifest_path}, "
                        f"line {line_idx}: {repr(line)}"
                    )

                image_path, ann_path = parts

                samples.append(
                    {
                        "image_path": image_path.strip(),
                        "ann_path": ann_path.strip(),
                    }
                )

        return samples

    def _check_files(self) -> None:
        missing = []

        for sample in self.samples:
            image_path = Path(sample["image_path"])
            ann_path = Path(sample["ann_path"])

            if not image_path.exists():
                missing.append(str(image_path))

            if not ann_path.exists():
                missing.append(str(ann_path))

        if missing:
            example = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"Found {len(missing)} missing files in {self.manifest_path}. "
                f"First examples:\n{example}"
            )

    @staticmethod
    def _load_annotation(path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        polygons = data.get("polygons", [])

        if len(polygons) == 0:
            raise RuntimeError(f"No polygons found in annotation: {path}")

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        image_path = Path(sample["image_path"])
        ann_path = Path(sample["ann_path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {image_path}") from e

        ann = self._load_annotation(ann_path)
        polygons = ann["polygons"]

        if self.transform is not None:
            image_tensor, polygons, meta = self.transform(image, polygons)
        else:
            image_tensor = image
            meta = {
                "orig_width": image.width,
                "orig_height": image.height,
            }

        sample_out = {
            "image": image_tensor,
            "polygons": polygons,
            "image_path": str(image_path),
            "ann_path": str(ann_path),
            "meta": meta,
        }

        if self.label_generator is not None:
            if hasattr(image_tensor, "shape"):
                image_height = int(image_tensor.shape[-2])
                image_width = int(image_tensor.shape[-1])
            else:
                image_width, image_height = image.size

            labels = self.label_generator(
                polygons=polygons,
                image_height=image_height,
                image_width=image_width,
            )

            sample_out.update(labels)

        return sample_out
