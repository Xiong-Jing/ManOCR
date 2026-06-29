from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def load_model_state(model: torch.nn.Module, path: str | Path, strict: bool = True, map_location="cpu") -> Dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=strict)
    return checkpoint
