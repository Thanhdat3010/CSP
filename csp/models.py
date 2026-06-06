# -*- coding: utf-8 -*-
"""
CSP Model Loaders.

Centralized wrappers for DINOv2 and MiDaS backbone initialization,
including their respective transforms.
"""

import logging

import torch
import torchvision.transforms as T

from . import config

logger = logging.getLogger("csp.models")


def load_dinov2(device: torch.device) -> torch.nn.Module:
    """Load DINOv2 ViT-S/14 and lock in evaluation mode.

    Args:
        device: Target device (cuda / cpu).

    Returns:
        Frozen DINOv2 model.
    """
    logger.info("Loading DINOv2 (%s) backbone...", config.DINO_MODEL)
    model = torch.hub.load(
        "facebookresearch/dinov2", config.DINO_MODEL, trust_repo=True
    ).to(device)
    model.eval()
    logger.info("DINOv2 initialized on %s", device)
    return model


def load_midas(device: torch.device):
    """Load MiDaS-Small depth estimator and its transforms.

    Args:
        device: Target device (cuda / cpu).

    Returns:
        Tuple of (model, transform).
    """
    logger.info("Loading MiDaS-Small depth estimator...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device)
    model.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    logger.info("MiDaS initialized on %s", device)
    return model, transforms


def get_dino_transforms() -> T.Compose:
    """Standard DINOv2 input transforms (resize → tensor → normalize)."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize((config.DINO_INPUT_SIZE, config.DINO_INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_dino_features(model: torch.nn.Module, image_tensor: torch.Tensor):
    """Extract both the CLS token and dense spatial patch features.

    Args:
        model: DINOv2 model.
        image_tensor: Tensor of shape (B, 3, H, W).

    Returns:
        cls_token: Tensor of shape (B, C) where C is feature dim.
        spatial_grid: Tensor of shape (B, H_grid, W_grid, C).
    """
    features = model.forward_features(image_tensor)
    cls_token = features["x_norm_clstoken"]
    patch_tokens = features["x_norm_patchtokens"]

    B, N, C = patch_tokens.shape
    H, W = image_tensor.shape[2:]
    grid_h = H // 14
    grid_w = W // 14

    spatial_grid = patch_tokens.reshape(B, grid_h, grid_w, C)
    return cls_token, spatial_grid

