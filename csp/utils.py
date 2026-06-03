# -*- coding: utf-8 -*-
"""
CSP Shared Utilities.

Deterministic seeding, device detection, logging setup, and filesystem helpers.
"""

import os
import random
import logging
import numpy as np
import torch

from . import config


def seed_everything(seed: int = config.SEED) -> None:
    """Lock all sources of randomness for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect CUDA/CPU and return the active device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure project-wide logging.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.

    Returns:
        Configured root logger for the 'csp' namespace.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger("csp")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it does not exist.

    Returns:
        The same path string for convenience.
    """
    os.makedirs(path, exist_ok=True)
    return path
