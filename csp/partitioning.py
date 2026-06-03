# -*- coding: utf-8 -*-
"""
CSP Phase 1A — Optimal Habitat Partitioning & Dataset Routing.

Corresponds to Cells 3-4 of the original notebook:
3. Optuna-driven Ward's linkage clustering (Davies-Bouldin minimization)
4. Multi-threaded file routing into habitat clusters
"""

import os
import shutil
import logging
import zipfile

import numpy as np
from glob import glob
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import davies_bouldin_score
import optuna
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .utils import seed_everything, ensure_dir

logger = logging.getLogger("csp.partitioning")


def optimize_clusters(
    V_FINAL: np.ndarray,
    n_trials: int = config.OPTUNA_N_TRIALS,
    k_min: int = config.OPTUNA_K_MIN,
    k_max: int = config.OPTUNA_K_MAX,
    seed: int = config.SEED,
):
    """Find the optimal number of habitat clusters using Optuna.

    Args:
        V_FINAL: PCA-compressed embedding matrix (N, D).
        n_trials: Number of Optuna trials.
        k_min: Minimum cluster count to search.
        k_max: Maximum cluster count to search.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: 'optimal_k', 'best_score', 'labels', 'unique_ids', 'counts'.
    """
    seed_everything(seed)

    # Stabilize floating-point noise for locked Optuna trajectory
    V_STABLE = np.round(V_FINAL, decimals=5)

    logger.info("Computing Ward's Linkage Matrix...")
    Z_matrix = linkage(V_STABLE, method="ward")

    def objective(trial):
        k = trial.suggest_int("k", k_min, k_max)
        labels = fcluster(Z_matrix, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            return float("inf")
        return davies_bouldin_score(V_STABLE, labels)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    optimal_k = study.best_params["k"]
    final_labels = fcluster(Z_matrix, t=optimal_k, criterion="maxclust")
    unique_ids, counts = np.unique(final_labels, return_counts=True)

    logger.info("CSP Optimization Complete.")
    logger.info("Optimal Habitats (k): %d | Best DB Score: %.4f", optimal_k, study.best_value)

    for uid, count in zip(unique_ids, counts):
        logger.info("  Habitat %02d: %d images", uid, count)

    return {
        "optimal_k": optimal_k,
        "best_score": study.best_value,
        "labels": final_labels,
        "unique_ids": unique_ids,
        "counts": counts,
    }


def compute_centroids(dino_l2: np.ndarray, labels: np.ndarray, unique_ids: np.ndarray):
    """Compute macro-environment centroids from uncompressed embeddings.

    Args:
        dino_l2: L2-normalized DINOv2 embeddings (N, 384).
        labels: Cluster assignment for each image.
        unique_ids: Array of unique cluster IDs.

    Returns:
        Dict mapping 'cluster_{id}' → centroid vector.
    """
    centroids = {}
    for uid in unique_ids:
        cluster_mask = labels == uid
        cluster_feats = dino_l2[cluster_mask]
        centroids[f"cluster_{uid}"] = np.mean(cluster_feats, axis=0)

    logger.info("Computed centroids for %d habitats.", len(centroids))
    return centroids


def route_dataset(
    image_paths: list,
    labels: np.ndarray,
    unique_ids: np.ndarray,
    data_dir: str,
    output_dir: str,
    max_workers: int = config.MAX_WORKERS,
):
    """Route image/label/mask triplets into cluster-specific directories.

    Args:
        image_paths: List of source image paths.
        labels: Cluster assignment for each image.
        unique_ids: Array of unique cluster IDs.
        data_dir: Root of the original dataset (for finding labels/masks).
        output_dir: Root output directory for the partitioned dataset.
        max_workers: Number of threads for parallel copying.

    Returns:
        Dict with keys: 'missing_labels', 'missing_masks'.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create directory structure
    for uid in unique_ids:
        ensure_dir(os.path.join(output_dir, "images", f"cluster_{uid}"))
        ensure_dir(os.path.join(output_dir, "labels", f"cluster_{uid}"))
        ensure_dir(os.path.join(output_dir, "masks", f"cluster_{uid}"))

    def route_file_group(path, label):
        img_out = os.path.join(output_dir, "images", f"cluster_{label}")
        lbl_out = os.path.join(output_dir, "labels", f"cluster_{label}")
        msk_out = os.path.join(output_dir, "masks", f"cluster_{label}")

        shutil.copy(path, img_out)

        parent_dir = os.path.dirname(os.path.dirname(path))
        base_name = os.path.splitext(os.path.basename(path))[0]

        label_src = os.path.join(parent_dir, "label", base_name + ".txt")
        mask_src = os.path.join(parent_dir, "mask", base_name + ".png")

        miss_lbl, miss_msk = 0, 0

        if os.path.exists(label_src):
            shutil.copy(label_src, lbl_out)
        else:
            miss_lbl = 1

        if os.path.exists(mask_src):
            shutil.copy(mask_src, msk_out)
        else:
            miss_msk = 1

        return miss_lbl, miss_msk

    missing_labels = 0
    missing_masks = 0
    tasks = []

    logger.info("Routing files to %d habitat clusters...", len(unique_ids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path, label in zip(image_paths, labels):
            tasks.append(executor.submit(route_file_group, path, label))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Parallel Routing Files"):
            lbl_miss, msk_miss = future.result()
            missing_labels += lbl_miss
            missing_masks += msk_miss

    logger.info("Routing complete. Missing: %d labels, %d masks.", missing_labels, missing_masks)

    return {"missing_labels": missing_labels, "missing_masks": missing_masks}


def run_partitioning(
    embeddings_dir: str,
    data_dir: str,
    output_dir: str,
    n_trials: int = config.OPTUNA_N_TRIALS,
    k_min: int = config.OPTUNA_K_MIN,
    k_max: int = config.OPTUNA_K_MAX,
    max_workers: int = config.MAX_WORKERS,
    seed: int = config.SEED,
):
    """Full partitioning pipeline: load embeddings → cluster → route → save.

    Args:
        embeddings_dir: Directory containing saved embeddings from Phase 1A.
        data_dir: Root of the original dataset.
        output_dir: Output directory for the partitioned dataset.
        n_trials, k_min, k_max: Optuna search parameters.
        max_workers: Threads for file routing.
        seed: Random seed.

    Returns:
        Dict with cluster results and routing stats.
    """
    seed_everything(seed)
    ensure_dir(output_dir)

    # Load pre-computed embeddings
    V_FINAL = np.load(os.path.join(embeddings_dir, "V_FINAL.npy"))
    dino_l2 = np.load(os.path.join(embeddings_dir, "dino_l2.npy"))
    image_paths = np.load(os.path.join(embeddings_dir, "image_paths.npy"), allow_pickle=True).tolist()

    # Optimize clusters
    cluster_result = optimize_clusters(V_FINAL, n_trials, k_min, k_max, seed)

    # Compute and save centroids
    centroids = compute_centroids(dino_l2, cluster_result["labels"], cluster_result["unique_ids"])
    centroid_path = os.path.join(output_dir, "habitat_centroids.npy")
    np.save(centroid_path, centroids)
    logger.info("Centroids saved to %s", centroid_path)

    # Route files
    partitioned_dir = os.path.join(output_dir, "CSP_Partitioned_Dataset")
    route_stats = route_dataset(
        image_paths,
        cluster_result["labels"],
        cluster_result["unique_ids"],
        data_dir,
        partitioned_dir,
        max_workers,
    )

    # Save labels
    np.save(os.path.join(output_dir, "cluster_labels.npy"), cluster_result["labels"])

    return {**cluster_result, **route_stats, "partitioned_dir": partitioned_dir}
