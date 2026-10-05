from contextlib import nullcontext

import torch
from torch.amp import GradScaler, autocast


def build_grad_scaler(device: torch.device, enabled: bool = True) -> GradScaler:
    return GradScaler("cuda", enabled=enabled and device.type == "cuda")


def autocast_context(device: torch.device, enabled: bool = True):
    if device.type == "cuda":
        return autocast("cuda", enabled=enabled)
    return nullcontext()
