from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class RecognitionDataset(Dataset):
    """
    Dataset for Manchu word image recognition.

    Manifest format:
        image_path<TAB>label
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Optional[Callable] = None,
        check_exists: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.samples = self._load_manifest(self.manifest_path)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in manifest: {self.manifest_path}")

        if check_exists:
            self._check_image_files()

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

                image_path, label = parts
                image_path = image_path.strip()
                label = label.strip()

                if image_path == "":
                    raise ValueError(f"Empty image path at line {line_idx}")

                if label == "":
                    raise ValueError(f"Empty label at line {line_idx}")

                samples.append(
                    {
                        "image_path": image_path,
                        "label": label,
                    }
                )

        return samples

    def _check_image_files(self) -> None:
        missing = []

        for sample in self.samples:
            image_path = Path(sample["image_path"])
            if not image_path.exists():
                missing.append(str(image_path))

        if missing:
            example = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"Found {len(missing)} missing images in {self.manifest_path}. "
                f"First examples:\n{example}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        image_path = Path(sample["image_path"])
        label = sample["label"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {image_path}") from e

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "image_path": str(image_path),
        }
