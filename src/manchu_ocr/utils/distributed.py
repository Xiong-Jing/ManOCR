import os

import torch


def is_dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_from_env(backend: str = "nccl") -> None:
    if is_dist_available_and_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend=backend)
