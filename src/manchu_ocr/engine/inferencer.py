import torch


class TorchInferencer:
    """Small inference helper that handles eval mode and no_grad."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, batch):
        self.model.eval()
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device, non_blocking=True)
        return self.model(batch)
