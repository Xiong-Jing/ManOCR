from typing import Dict, List

import torch
import numpy as np


class DetCollate:
    """
    Collate function for detection.

    If DB label maps exist, stack them into tensors.
    """

    def __call__(self, batch: List[Dict]) -> Dict:
        images = torch.stack([item["image"] for item in batch], dim=0)

        polygons = [item["polygons"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        ann_paths = [item["ann_path"] for item in batch]
        metas = [item["meta"] for item in batch]

        output = {
            "images": images,
            "polygons": polygons,
            "image_paths": image_paths,
            "ann_paths": ann_paths,
            "metas": metas,
        }

        map_keys = [
            "prob_map",
            "thresh_map",
            "thresh_mask",
            "training_mask",
        ]

        for key in map_keys:
            if key in batch[0]:
                values = [
                    torch.from_numpy(item[key]).float()
                    if isinstance(item[key], np.ndarray)
                    else item[key].float()
                    for item in batch
                ]
                output[key] = torch.stack(values, dim=0).unsqueeze(1)

        if "num_valid_polygons" in batch[0]:
            values = [
                torch.from_numpy(item["num_valid_polygons"]).long()
                if isinstance(item["num_valid_polygons"], np.ndarray)
                else item["num_valid_polygons"].long()
                for item in batch
            ]
            output["num_valid_polygons"] = torch.stack(values, dim=0).view(-1)

        return output
