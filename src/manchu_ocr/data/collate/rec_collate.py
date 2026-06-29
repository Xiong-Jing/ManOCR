from typing import Dict, List

import torch


class RecCollate:
    """
    Collate function for recognition training.

    Output:
        images: Tensor [B, C, H, W]
        labels: original string labels
        targets: concatenated CTC targets
        target_lengths: target lengths for CTC
        image_paths: image paths
    """

    def __init__(self, label_converter):
        self.label_converter = label_converter

    def __call__(self, batch: List[Dict]) -> Dict:
        images = torch.stack([item["image"] for item in batch], dim=0)
        labels = [item["label"] for item in batch]
        image_paths = [item["image_path"] for item in batch]

        targets, target_lengths = self.label_converter.encode(labels)

        return {
            "images": images,
            "labels": labels,
            "targets": targets,
            "target_lengths": target_lengths,
            "image_paths": image_paths,
        }
