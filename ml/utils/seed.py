"""Reproducibility helpers: global seeding and device selection."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """Return a valid torch.device, falling back to CPU if CUDA is absent."""
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    return torch.device("cpu")
