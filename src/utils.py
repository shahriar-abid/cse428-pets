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


def on_kaggle() -> bool:
    """True when running inside a Kaggle kernel session."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or os.path.isdir("/kaggle")


KAGGLE_REPO_DIR = "/kaggle/working/cse428-pets"


def resolve_output_dir(rel_dir: str) -> str:
    """Return where checkpoints/reports should be written so Kaggle persists them
    into the notebook output bucket.

    Kaggle's output capture is scoped to the git-clone snapshot
    (/kaggle/working/<repo>) plus whatever a session adds inside it, but NOT to
    arbitrary siblings of /kaggle/working. So on Kaggle outputs must live INSIDE
    the cloned repo (os.getcwd() after the clone cell) to be captured for resume
    / groupmate handoff. Locally we keep the plain relative path."""
    if on_kaggle():
        deploy_dir = os.environ.get("CSE428_DEPLOY_DIR") or os.getcwd()
        return os.path.join(deploy_dir, os.path.basename(rel_dir))
    return rel_dir


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
