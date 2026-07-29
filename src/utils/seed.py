"""Single entry point for reproducible randomness across the project.

Every training/splitting script must call ``set_global_seed`` once, using the
seed from ``config/config.yaml`` (``settings.random_seed``), never a
hardcoded literal, so a fresh clone reproduces the recorded results.
"""
from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
