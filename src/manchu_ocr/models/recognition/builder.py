from manchu_ocr.models.registry import RECOGNIZERS

# trigger registration
from manchu_ocr.models.recognition.svtr_official import SVTROfficialRecognizer  # noqa: F401


def build_recognition_model(cfg: dict, num_classes: int):
    """
    Build recognition model from config.

    Registry-style configs:
        model:
          name: "SVTROfficialRecognizer"
          ...

    Legacy configs are still supported through SVTRRecognizer.
    """
    model_cfg = cfg["model"].copy()
    model_name = model_cfg.get("name", "")

    registry_model_names = {
        "SVTROfficialRecognizer",
    }

    if model_name in registry_model_names:
        model_cfg["num_classes"] = num_classes
        return RECOGNIZERS.build(model_cfg)

    # legacy fallback: keep old svtr_baseline / svtr_dab / svtr_lortho configs usable
    from manchu_ocr.models.recognition.svtr import SVTRRecognizer

    return SVTRRecognizer(
        num_classes=num_classes,
        in_channels=int(model_cfg["in_channels"]),
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        sequence_axis=cfg["data"]["sequence_axis"],
        use_diacritic_branch=bool(model_cfg.get("use_diacritic_branch", False)),
    )
