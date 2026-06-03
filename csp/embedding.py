# -*- coding: utf-8 -*-
"""
CSP Phase 1A — Semantic Embedding & Compression.

Corresponds to Cells 1-2 of the original notebook:
1. Dataset download from Google Drive (via gdown)
2. DINOv2 feature extraction
3. L2 normalization + StandardScaler + PCA compression
"""

import os
import gc
import shutil
import logging
import zipfile

import cv2
import gdown
import numpy as np
import torch
from glob import glob
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .utils import seed_everything, get_device, ensure_dir
from .models import load_dinov2
from .datasets import RawSemanticDataset, collate_filter_none

logger = logging.getLogger("csp.embedding")


def download_dataset(drive_file_id: str, workspace_dir: str) -> str:
    """Download and extract the dataset from Google Drive.

    Args:
        drive_file_id: Google Drive file ID for the dataset zip.
        workspace_dir: Local directory to extract into.

    Returns:
        Path to the workspace directory containing extracted data.
    """
    zip_dest = os.path.join(os.path.dirname(workspace_dir), "dataset.zip")

    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    ensure_dir(workspace_dir)

    logger.info("Downloading dataset from Google Drive (ID: %s)...", drive_file_id)
    gdown.download(id=drive_file_id, output=zip_dest, quiet=False)

    logger.info("Extracting dataset to %s...", workspace_dir)
    with zipfile.ZipFile(zip_dest, "r") as zf:
        zf.extractall(workspace_dir)

    os.remove(zip_dest)
    logger.info("Dataset extracted successfully.")
    return workspace_dir


def find_image_paths(workspace_dir: str, subset: str = "train/camo/image") -> list:
    """Locate ground-truth images inside the workspace.

    Args:
        workspace_dir: Root of the extracted dataset.
        subset: Relative path pattern to match (e.g., 'train/camo/image').

    Returns:
        Sorted list of absolute image paths.
    """
    all_paths = sorted(
        glob(os.path.join(workspace_dir, "**", "image", "*.jpg"), recursive=True)
        + glob(os.path.join(workspace_dir, "**", "image", "*.png"), recursive=True)
    )
    # Filter to the target subset
    filtered = [p for p in all_paths if subset.replace("/", os.sep) in p
                or subset.replace("\\", "/") in p.replace("\\", "/")]
    if not filtered:
        # Fallback: return all images
        filtered = all_paths
    logger.info("Located %d images for embedding (subset: %s)", len(filtered), subset)
    return filtered


def extract_embeddings(
    data_dir: str,
    output_dir: str,
    drive_file_id: str = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
    pca_variance: float = config.PCA_VARIANCE_RATIO,
):
    """Run the full embedding pipeline: download → extract → DINOv2 → PCA.

    Args:
        data_dir: Path to the dataset root. If it doesn't exist and
                  drive_file_id is provided, the dataset will be downloaded.
        output_dir: Directory to save embedding outputs.
        drive_file_id: Google Drive file ID (optional, for auto-download).
        batch_size: Batch size for DINOv2 inference.
        num_workers: Number of DataLoader workers.
        seed: Random seed.
        pca_variance: PCA variance retention ratio.

    Returns:
        Dict with keys: 'V_FINAL' (PCA embeddings), 'dino_l2' (normalized
        embeddings), 'image_paths' (list of paths).
    """
    seed_everything(seed)
    device = get_device()
    ensure_dir(output_dir)

    # --- Download if needed ---
    if drive_file_id and not os.path.exists(data_dir):
        download_dataset(drive_file_id, data_dir)

    # --- Locate images ---
    img_files = find_image_paths(data_dir)
    if not img_files:
        raise FileNotFoundError(f"No images found in {data_dir}")

    # --- DINOv2 extraction ---
    dino = load_dinov2(device)

    dataset = RawSemanticDataset(img_files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_filter_none,
        pin_memory=True,
    )

    all_feats = {"dino": [], "paths": []}

    logger.info("Extracting DINOv2 embeddings (%d images)...", len(img_files))
    with torch.no_grad():
        for img_pt, paths in tqdm(dataloader, desc="Extracting DINOv2 Embeddings"):
            img_pt = img_pt.to(device, non_blocking=True)
            dino_out = dino(img_pt).cpu().numpy()
            all_feats["dino"].append(dino_out)
            all_feats["paths"].extend(paths)

    # --- Normalize & Compress ---
    V_DINO = np.vstack(all_feats["dino"])
    dino_l2 = normalize(V_DINO, norm="l2", axis=1)
    v_dino_scaled = StandardScaler().fit_transform(dino_l2)

    pca = PCA(n_components=pca_variance, random_state=seed)
    V_FINAL = pca.fit_transform(v_dino_scaled)

    # Cleanup GPU
    del dino
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(
        "Embedding complete. Original: %d dims → Compressed: %d dims (%.0f%% variance)",
        config.DINO_FEATURE_DIM,
        V_FINAL.shape[1],
        pca_variance * 100,
    )

    # --- Save outputs ---
    np.save(os.path.join(output_dir, "V_FINAL.npy"), V_FINAL)
    np.save(os.path.join(output_dir, "dino_l2.npy"), dino_l2)
    np.save(os.path.join(output_dir, "image_paths.npy"), np.array(all_feats["paths"]))

    logger.info("Embeddings saved to %s", output_dir)

    return {
        "V_FINAL": V_FINAL,
        "dino_l2": dino_l2,
        "image_paths": all_feats["paths"],
    }
