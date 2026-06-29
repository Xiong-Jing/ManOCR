from pathlib import Path
from typing import Dict, List

from PIL import Image
from torch.utils.data import Dataset


class FullPipelineDataset(Dataset):
    """Image-only dataset for full OCR inference."""

    def __init__(self, image_paths: str | Path | List[str | Path], check_exists: bool = True):
        if isinstance(image_paths, (str, Path)) and Path(image_paths).is_file():
            with Path(image_paths).open("r", encoding="utf-8") as f:
                self.image_paths = [Path(line.strip().split("\t")[0]) for line in f if line.strip()]
        elif isinstance(image_paths, (str, Path)) and Path(image_paths).is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            self.image_paths = [p for p in sorted(Path(image_paths).rglob("*")) if p.suffix.lower() in exts]
        else:
            self.image_paths = [Path(p) for p in image_paths]

        if check_exists:
            missing = [str(p) for p in self.image_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} images. First examples: {missing[:5]}")

        if len(self.image_paths) == 0:
            raise RuntimeError("No images found for full OCR inference.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return {"image": image, "image_path": str(image_path)}
