# -*- coding: utf-8 -*-
"""
CSP Dataset Classes.

All PyTorch Dataset definitions used across the pipeline phases,
consolidated from Cells 2, 6, 7, and 8.
"""

import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


def collate_filter_none(batch):
    """Collate function that filters out None entries."""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class RawSemanticDataset(Dataset):
    """Dataset for extracting DINOv2 embeddings from raw images (Cell 2).

    Returns (tensor, path) pairs where tensor is a normalized 224×224 image.
    """

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (config.DINO_INPUT_SIZE, config.DINO_INPUT_SIZE))

        img_pt = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0
        return img_pt, path


class EnvDataset(Dataset):
    """Dataset for MiDaS depth extraction per image (Cell 6 Phase A).

    Returns (tensor, path, is_valid, height, width).
    """

    def __init__(self, registry, transform):
        self.img_paths = list(registry.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv2.imread(path)
        if img is None:
            return torch.zeros(3, 256, 256), path, False, 0, 0

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (256, 256))

        tensor = self.transform(img_rgb).squeeze(0)
        return tensor, path, True, h, w


class PatchDataset(Dataset):
    """Dataset for DINOv2 object profiling on cropped patches (Cell 6 Phase B).

    Returns (tensor, object_id).
    """

    def __init__(self, patch_registry, transform):
        self.patch_registry = patch_registry
        self.transform = transform

    def __len__(self):
        return len(self.patch_registry)

    def __getitem__(self, idx):
        data = self.patch_registry[idx]
        return self.transform(data["patch"]), data["obj_id"]


class BackgroundDataset(Dataset):
    """Dataset for background environment cataloging (Cell 7).

    Returns (tensor, path, is_valid, height, width).
    """

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            return (
                torch.zeros((3, config.DINO_INPUT_SIZE, config.DINO_INPUT_SIZE)),
                img_path,
                False,
                0,
                0,
            )

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb)
        return tensor, img_path, True, h, w


class AtomicSynthesisDataset(Dataset):
    """Dataset for task-centric synthesis (Cell 8).

    Each item is one (background, object) atomic task.
    Returns (tensor, bg_path, obj_json_str, is_valid, height, width, bg_meta_json).
    """

    def __init__(self, tasks, transform):
        self.tasks = tasks
        self.transform = transform

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        bg_path = task["bg_path"]

        img = cv2.imread(bg_path)
        if img is None:
            return torch.zeros(3, 256, 256), bg_path, json.dumps({}), False, 0, 0, json.dumps({})

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (256, 256))
        tensor = self.transform(img_rgb).squeeze(0)

        obj_json = json.dumps(task["obj_data"])
        bg_meta_json = json.dumps(task.get("bg_meta", {}))
        return tensor, bg_path, obj_json, True, h, w, bg_meta_json
