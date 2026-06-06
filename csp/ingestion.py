# -*- coding: utf-8 -*-
"""
CSP Phase 2 — Universal Background Ingestion.

Corresponds to Cell 7 of the original notebook:
1. Scan partitioned + external images for empty spaces
2. Distance Transform to find R_avail
3. Cosine-similarity habitat matching for external images
4. Build environment_catalog.json
"""

import os
import gc
import json
import shutil
import logging
import pickle

import cv2
import numpy as np
import torch
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .utils import seed_everything, get_device, ensure_dir
from .models import load_dinov2, get_dino_transforms, extract_dino_features
from .datasets import BackgroundDataset
from .physics import get_patch_spherical_harmonics

logger = logging.getLogger("csp.ingestion")


def ingest_backgrounds(
    partitioned_dir: str,
    centroids_path: str,
    new_data: str,
    output_path: str,
    rho_max: float = config.RHO_MAX,
    sim_threshold: float = config.SIMILARITY_THRESHOLD,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
):
    """Catalog all available background environments for synthesis.

    Args:
        partitioned_dir: Root of CSP_Partitioned_Dataset.
        centroids_path: Path to habitat_centroids.npy.
        new_data: Path to custom background directory (containing image/ and label/).
        output_path: Output path for environment_catalog.json.
        rho_max: Maximum background saturation threshold.
        sim_threshold: Minimum cosine similarity for habitat matching.
        batch_size: Batch size for DINOv2 inference.
        num_workers: DataLoader workers.
        seed: Random seed.

    Returns:
        The background catalog dict (also saved to output_path).
    """
    seed_everything(seed)
    device = get_device()
    output_dir = os.path.dirname(output_path)
    ensure_dir(output_dir)

    # Locate and load PCA and scaler from Step 1
    embeddings_dir = os.path.join(output_dir, "embeddings")
    if not os.path.exists(os.path.join(embeddings_dir, "pca.pkl")):
        embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(partitioned_dir)), "embeddings")
    if not os.path.exists(os.path.join(embeddings_dir, "pca.pkl")):
        embeddings_dir = "./outputs/embeddings"

    pca_path = os.path.join(embeddings_dir, "pca.pkl")
    scaler_path = os.path.join(embeddings_dir, "scaler.pkl")

    logger.info("Loading PCA and Scaler from %s...", embeddings_dir)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    background_features_dir = os.path.join(output_dir, "background_features")
    ensure_dir(background_features_dir)

    # Load DINOv2
    dino = load_dinov2(device)
    dino_transforms = get_dino_transforms()

    # Load centroids
    logger.info("Loading habitat centroids from %s...", centroids_path)
    habitat_centroids = np.load(centroids_path, allow_pickle=True).item()
    logger.info("Loaded anchors for %d habitats.", len(habitat_centroids))

    # Collect all image paths
    reclaimed_source = os.path.join(partitioned_dir, "images")
    external_source = os.path.join(new_data, "image")

    all_image_paths = sorted(
        glob(os.path.join(reclaimed_source, "**", "*.jpg"), recursive=True)
        + glob(os.path.join(reclaimed_source, "**", "*.png"), recursive=True)
        + glob(os.path.join(external_source, "*.jpg"))
        + glob(os.path.join(external_source, "*.png"))
    )
    logger.info("Scanning %d candidate backgrounds...", len(all_image_paths))

    dataset = BackgroundDataset(all_image_paths, dino_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    background_catalog = {}

    with torch.no_grad():
        for batch_tensors, batch_paths, batch_valid, batch_h, batch_w in tqdm(
            dataloader, desc="Cataloging Backgrounds"
        ):
            # DINOv2 for all valid images in the batch
            valid_indices = [idx for idx, valid in enumerate(batch_valid) if valid]

            batch_cls = {}
            batch_spatial = {}
            if valid_indices:
                valid_tensors = torch.stack([batch_tensors[idx] for idx in valid_indices]).to(device)
                cls_tokens, spatial_grids = extract_dino_features(dino, valid_tensors)
                cls_tokens = cls_tokens.cpu().numpy()
                spatial_grids = spatial_grids.cpu().numpy()

                for j, orig_idx in enumerate(valid_indices):
                    batch_cls[orig_idx] = cls_tokens[j]
                    batch_spatial[orig_idx] = spatial_grids[j]

            # Process each image
            for i in range(len(batch_paths)):
                if not batch_valid[i]:
                    continue

                img_path = batch_paths[i]
                img_name = os.path.basename(img_path)
                bg_base_name = os.path.splitext(img_name)[0]
                orig_H, orig_W = batch_h[i].item(), batch_w[i].item()

                # Find label path
                if img_path.startswith(external_source):
                    lbl_path = os.path.join(new_data, "label", f"{bg_base_name}.txt")
                else:
                    lbl_path = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
                    lbl_path = os.path.splitext(lbl_path)[0] + ".txt"

                # Build occupancy mask at native resolution
                existing_mask = np.zeros((orig_H, orig_W), dtype=np.uint8)

                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as f:
                        for line in f.readlines():
                            try:
                                bbox = [float(x) for x in line.strip().split()][:5]
                                _, x_c, y_c, bw, bh = bbox
                                x1 = int((x_c - bw / 2) * orig_W)
                                x2 = int((x_c + bw / 2) * orig_W)
                                y1 = int((y_c - bh / 2) * orig_H)
                                y2 = int((y_c + bh / 2) * orig_H)
                                cv2.rectangle(existing_mask, (x1, y1), (x2, y2), 255, -1)
                            except Exception:
                                continue

                saturation = np.count_nonzero(existing_mask) / (orig_H * orig_W)
                if saturation >= rho_max:
                    continue

                empty_space = np.uint8((existing_mask == 0) * 255)
                dist_transform = cv2.distanceTransform(empty_space, cv2.DIST_L2, 5)
                normalized_r_avail = float(np.max(dist_transform)) / orig_W

                if normalized_r_avail < 0.02:
                    continue

                # Global Spherical Harmonics calculation avoiding cluttered areas
                sh_mask = np.uint8((dist_transform > 5) * 255)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                global_sh = get_patch_spherical_harmonics(img, mask=sh_mask)

                # Classify habitat
                best_match_cluster = None
                best_sim = 1.0

                if "CSP_Partitioned_Dataset" in img_path or "Partitioned" in img_path:
                    best_match_cluster = os.path.basename(os.path.dirname(img_path))
                else:
                    if i not in batch_cls:
                        continue
                    bg_feat = batch_cls[i]
                    best_sim = -1
                    for cluster_name, centroid in habitat_centroids.items():
                        sim = cosine_similarity(
                            bg_feat.reshape(1, -1), centroid.reshape(1, -1)
                        )[0][0]
                        if sim > best_sim:
                            best_sim, best_match_cluster = sim, cluster_name
                    if best_sim < sim_threshold:
                        continue

                # Compress and save spatial grid
                if i not in batch_spatial:
                    continue
                grid = batch_spatial[i]  # Shape: (H_grid, W_grid, C)
                H_grid, W_grid, C = grid.shape
                grid_flat = grid.reshape(-1, C)
                grid_l2 = normalize(grid_flat, norm="l2", axis=1)
                grid_scaled = scaler.transform(grid_l2)
                grid_pca = pca.transform(grid_scaled)
                grid_compressed = grid_pca.reshape(H_grid, W_grid, -1)

                bg_feat_path = os.path.join(background_features_dir, f"{bg_base_name}.npy")
                np.save(bg_feat_path, grid_compressed)

                if best_match_cluster not in background_catalog:
                    background_catalog[best_match_cluster] = {}

                background_catalog[best_match_cluster][img_name] = {
                    "path": img_path,
                    "semantic_confidence": float(best_sim),
                    "saturation": float(saturation),
                    "R_avail_norm": float(normalized_r_avail),
                    "orig_w": orig_W,
                    "orig_h": orig_H,
                    "global_sh": global_sh,
                    "feature_path": bg_feat_path,
                }

    with open(output_path, "w") as f:
        json.dump(background_catalog, f, indent=4)

    # Cleanup
    del dino
    torch.cuda.empty_cache()
    gc.collect()

    total_bgs = sum(len(v) for v in background_catalog.values())
    logger.info("Cataloged %d backgrounds across %d habitats → %s",
                total_bgs, len(background_catalog), output_path)

    return background_catalog
