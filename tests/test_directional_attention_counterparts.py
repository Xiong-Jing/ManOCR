from __future__ import annotations

from pathlib import Path

import torch

from manchu_ocr.models.detection.necks.db_fpn import DBFPN
from manchu_ocr.models.detection.modules.directional_attention import (
    build_direction_module,
)
from manchu_ocr.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUNTERPARTS = {
    "none": "configs/detection/dbnetpp_official_baseline.yaml",
    "strip_pooling": "configs/detection/dbnetpp_strip_pooling.yaml",
    "coordinate_attention": "configs/detection/dbnetpp_coordinate_attention.yaml",
    "horizontal_strip": "configs/detection/dbnetpp_horizontal_strip.yaml",
    "vertical_strip": "configs/detection/dbnetpp_vertical_strip.yaml",
    "vsaa": "configs/detection/dbnetpp_vsaa.yaml",
}


def test_all_direction_modules_preserve_shape():
    x = torch.randn(2, 32, 16, 12)
    for name in COUNTERPARTS:
        module = build_direction_module(
            name,
            channels=32,
            strip_pool_sizes=((8, 6), (6, 8)),
            vsaa_vertical_kernel=5,
            vsaa_horizontal_kernel=3,
        ).eval()
        with torch.no_grad():
            output = module(x)
        assert output.shape == x.shape, name


def test_legacy_use_vsaa_keeps_checkpoint_parameter_names():
    legacy_none = DBFPN(
        in_channels=[64, 128, 256, 512],
        inner_channels=128,
        out_channels=256,
        use_asf=True,
        use_vsaa=False,
    )
    explicit_none = DBFPN(
        in_channels=[64, 128, 256, 512],
        inner_channels=128,
        out_channels=256,
        use_asf=True,
        use_vsaa=False,
        direction_module="none",
    )
    legacy = DBFPN(
        in_channels=[64, 128, 256, 512],
        inner_channels=128,
        out_channels=256,
        use_asf=True,
        use_vsaa=True,
    )
    explicit = DBFPN(
        in_channels=[64, 128, 256, 512],
        inner_channels=128,
        out_channels=256,
        use_asf=True,
        use_vsaa=True,
        direction_module="vsaa",
    )
    assert legacy_none.state_dict().keys() == explicit_none.state_dict().keys()
    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    legacy_vsaa_keys = {
        key for key in legacy.state_dict() if key.startswith("vsaa.")
    }
    assert legacy_vsaa_keys


def test_counterpart_configs_hold_controlled_variables_fixed():
    configs = {
        module_name: load_yaml(PROJECT_ROOT / config_path)
        for module_name, config_path in COUNTERPARTS.items()
    }
    baseline = configs["none"]
    vsaa = configs["vsaa"]

    for module_name, cfg in configs.items():
        assert cfg["experiment"]["paths_config"] == baseline["experiment"]["paths_config"]
        assert cfg["data"] == baseline["data"], module_name
        assert cfg["model"]["backbone"] == baseline["model"]["backbone"], module_name
        assert cfg["model"]["head"] == baseline["model"]["head"], module_name
        assert cfg["loss"] == baseline["loss"], module_name
        assert cfg["train"] == baseline["train"], module_name

        neck = cfg["model"]["neck"]
        assert neck["name"] == "DBFPN"
        assert neck["inner_channels"] == 128
        assert neck["out_channels"] == 256
        assert neck["use_asf"] is True
        assert neck.get("direction_module", "vsaa" if neck["use_vsaa"] else "none") == module_name
        assert cfg["train"]["epochs"] == 100
        assert cfg["train"]["val_interval"] == 1
        assert cfg["eval"]["iou_thresh"] == 0.75

        assert "eval_degradation" not in cfg
        assert cfg["eval"] == baseline["eval"], module_name


def test_counterpart_plan_lists_exactly_the_six_configs():
    plan = load_yaml(
        PROJECT_ROOT / "configs" / "experiments" / "det_vsaa_counterparts.yaml"
    )
    planned = {item["direction_module"]: item["config"] for item in plan["experiments"]}
    assert planned == COUNTERPARTS
    assert plan["controlled_variables"]["optimizer"] == "AdamW"
    assert plan["validation_policy"]["every_epoch"] is True
