from manchu_ocr.models.registry import DETECTORS

# import detector to trigger registration
from manchu_ocr.models.detection.dbnetpp import DBNetPP  # noqa: F401


def build_detection_model(cfg: dict):
    """
    Build detection model from config.

    Expected:
        cfg["model"] = {
            "name": "DBNetPP",
            "backbone": {...},
            "neck": {...},
            "head": {...}
        }
    """
    model_cfg = cfg["model"].copy()
    return DETECTORS.build(model_cfg)
