"""Reproducibility and device helpers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_device(device: torch.device) -> torch.device:
    """Fail fast on GPUs PyTorch cannot run kernels on (e.g. Kaggle P100 vs
    a PyTorch build supporting sm_70+), with an actionable message."""
    if device.type != "cuda":
        return device
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 0):
        name = torch.cuda.get_device_name(device)
        raise RuntimeError(
            f"{name} has CUDA capability sm_{capability[0]}{capability[1]}, but "
            "this PyTorch build only ships kernels for sm_70+. On Kaggle, switch "
            "the accelerator from P100 to 'GPU T4 x2' (session options)."
        )
    return device
