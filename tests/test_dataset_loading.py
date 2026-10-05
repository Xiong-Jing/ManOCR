from pathlib import Path

from PIL import Image

from manchu_ocr.data.datasets.recognition_dataset import RecognitionDataset


def test_recognition_dataset_loads_manifest(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 16), color=(255, 255, 255)).save(image_path)

    manifest = tmp_path / "train.txt"
    manifest.write_text(f"{image_path}\tab\n", encoding="utf-8")

    dataset = RecognitionDataset(manifest, check_exists=True)
    item = dataset[0]

    assert item["label"] == "ab"
    assert item["image_path"] == str(image_path)
