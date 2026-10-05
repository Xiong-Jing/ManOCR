import torch
import torch.nn as nn

from manchu_ocr.models.registry import BACKBONES, DETECTORS, HEADS, NECKS

# import modules to trigger registry registration
from manchu_ocr.models.detection.backbones.resnet import ResNetBackbone  # noqa: F401
from manchu_ocr.models.detection.heads.db_head import DBHead  # noqa: F401
from manchu_ocr.models.detection.necks.db_fpn import DBFPN  # noqa: F401


@DETECTORS.register("DBNetPP")
class DBNetPP(nn.Module):
    """
    Official-style DBNet++ detector.

    Modules:
        backbone
        neck
        head

    Current supported:
        backbone: ResNetBackbone
        neck: DBFPN
        head: DBHead
    """

    def __init__(
        self,
        backbone: dict,
        neck: dict,
        head: dict,
    ):
        super().__init__()

        self.backbone = BACKBONES.build(backbone)

        neck = neck.copy()

        if "in_channels" not in neck:
            neck["in_channels"] = self.backbone.out_channels

        self.neck = NECKS.build(neck)
        self.head = HEADS.build(head)

    def forward(self, images: torch.Tensor) -> dict:
        features = self.backbone(images)
        fused = self.neck(features)
        outputs = self.head(fused)
        return outputs
