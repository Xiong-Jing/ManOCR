from manchu_ocr.models.registry import DETECTORS

# import detector to trigger registration
from manchu_ocr.models.detection.craft import CRAFTDetector  # noqa: F401
from manchu_ocr.models.detection.dbnet import DBNet  # noqa: F401
from manchu_ocr.models.detection.dbnetpp import DBNetPP  # noqa: F401
from manchu_ocr.models.detection.east import EASTDetector  # noqa: F401
from manchu_ocr.models.detection.hisam import HiSAMDetector  # noqa: F401
from manchu_ocr.models.detection.ppocrv5 import PPOCRv5Detector  # noqa: F401


def build_detection_model(cfg: dict):
    """
    Build detection model from config.

    Expected:
        cfg["model"] = {
            "name": "DBNetPP", "DBNet", "EASTDetector", "CRAFTDetector",
                    "HiSAMDetector", or "PPOCRv5Detector",
            "backbone": {...},
            "neck": {...},
            "head": {...}
        }
    """
    model_cfg = cfg["model"].copy()
    return DETECTORS.build(model_cfg)
