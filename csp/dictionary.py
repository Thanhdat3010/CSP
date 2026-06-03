# -*- coding: utf-8 -*-
"""
CSP Phase 1B — Dual-Threaded Latent Affordance Dictionary.

Corresponds to Cell 6 of the original notebook:
1. MiDaS batched depth extraction (Phase A)
2. DINOv2 batched object profiling (Phase B)
3. Isolation Forest anomaly sieve
4. JSON dictionary assembly
"""

import os
import gc
import json
import logging

import cv2
import numpy as np
import torch
from glob import glob
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .utils import seed_everything, get_device, ensure_dir
from .models import load_dinov2, load_midas, get_dino_transforms
from .datasets import EnvDataset, PatchDataset
from .physics import get_topographical_features, get_spherical_harmonics, get_adaptive_solidity

logger = logging.getLogger("csp.dictionary")


def _build_image_registry(partitioned_dir: str) -> dict:
    """Build a registry of all images with their labels and masks.

    Args:
        partitioned_dir: Root of CSP_Partitioned_Dataset.

    Returns:
        Dict mapping image_path → {cluster, bboxes, mask_path, img_name}.
    """
    cluster_img_root = os.path.join(partitioned_dir, "images")
    cluster_lbl_root = os.path.join(partitioned_dir, "labels")
    cluster_msk_root = os.path.join(partitioned_dir, "masks")

    registry = {}
    cluster_dirs = sorted([d for d in os.listdir(cluster_img_root) if d.startswith("cluster_")])

    for cluster_name in cluster_dirs:
        img_dir = os.path.join(cluster_img_root, cluster_name)
        lbl_dir = os.path.join(cluster_lbl_root, cluster_name)
        msk_dir = os.path.join(cluster_msk_root, cluster_name)

        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.endswith((".jpg", ".png")):
                continue

            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, base_name + ".txt")

            if not os.path.exists(lbl_path):
                continue

            # Find mask
            mask_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                test_path = os.path.join(msk_dir, base_name + ext)
                if os.path.exists(test_path):
                    mask_path = test_path
                    break

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            valid_bboxes = [
                [float(x) for x in line.strip().split()][:5]
                for line in lines
                if len(line.strip().split()) >= 5
            ]

            if valid_bboxes:
                registry[img_path] = {
                    "cluster": cluster_name,
                    "bboxes": valid_bboxes,
                    "mask_path": mask_path,
                    "img_name": img_name,
                }

    logger.info("Image registry built: %d images with valid annotations.", len(registry))
    return registry


def build_dictionary(
    partitioned_dir: str,
    output_path: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    contamination: float = config.ISOLATION_CONTAMINATION,
    seed: int = config.SEED,
):
    """Build the Latent Affordance Dictionary.

    Two-phase pipeline:
      A) MiDaS depth → topography, SH, solidity extraction per object
      B) DINOv2 → object semantic embeddings for anomaly detection

    Args:
        partitioned_dir: Root of CSP_Partitioned_Dataset.
        output_path: Output JSON path for the dictionary.
        batch_size: Batch size for model inference.
        num_workers: DataLoader workers.
        contamination: Isolation Forest contamination rate.
        seed: Random seed.

    Returns:
        The latent dictionary (also saved to output_path).
    """
    seed_everything(seed)
    device = get_device()
    ensure_dir(os.path.dirname(output_path))

    # Build image registry
    registry = _build_image_registry(partitioned_dir)

    # Load models
    midas, midas_transforms = load_midas(device)
    dino = load_dinov2(device)
    dino_transforms = get_dino_transforms()

    # --- Phase A: MiDaS Topography Extraction ---
    logger.info("Phase A: Extracting topography via MiDaS batching...")
    env_loader = DataLoader(
        EnvDataset(registry, midas_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    patch_registry = []
    obj_metadata = {}
    obj_counter = 0

    with torch.no_grad():
        for batch_tensors, batch_paths, batch_valid, batch_h, batch_w in tqdm(
            env_loader, desc="Topography Extraction"
        ):
            batch_tensors = batch_tensors.to(device)
            depth_preds = midas(batch_tensors)

            for i in range(len(batch_paths)):
                if not batch_valid[i]:
                    continue

                img_path = batch_paths[i]
                H, W = batch_h[i].item(), batch_w[i].item()
                reg_info = registry[img_path]

                depth_map = torch.nn.functional.interpolate(
                    depth_preds[i].unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

                if reg_info["mask_path"]:
                    full_mask = cv2.imread(reg_info["mask_path"], cv2.IMREAD_GRAYSCALE)
                    _, full_mask = cv2.threshold(full_mask, 127, 255, cv2.THRESH_BINARY)
                    if full_mask.shape[:2] != (H, W):
                        full_mask = cv2.resize(full_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    full_mask = np.zeros((H, W), dtype=np.uint8)

                for bbox in reg_info["bboxes"]:
                    class_id = int(bbox[0])
                    _, x_c, y_c, bw, bh = bbox
                    area = bw * bh

                    x1 = max(0, int((x_c - bw / 2) * W))
                    x2 = min(W, int((x_c + bw / 2) * W))
                    y1 = max(0, int((y_c - bh / 2) * H))
                    y2 = min(H, int((y_c + bh / 2) * H))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    grad_D, surface_normal = get_topographical_features(depth_map, bbox, W, H)
                    c_sh = get_spherical_harmonics(gray_img, bbox, W, H, mask=full_mask)

                    obj_mask = np.zeros((H, W), dtype=np.uint8)
                    if reg_info["mask_path"]:
                        obj_mask[y1:y2, x1:x2] = full_mask[y1:y2, x1:x2]
                    else:
                        cv2.rectangle(obj_mask, (x1, y1), (x2, y2), 255, -1)

                    solidity = get_adaptive_solidity(obj_mask[y1:y2, x1:x2])
                    pure_obj = cv2.bitwise_and(img, img, mask=obj_mask)
                    cropped = pure_obj[y1:y2, x1:x2]

                    if cropped.size > 0 and cropped.shape[0] > 10 and cropped.shape[1] > 10:
                        obj_id = f"{reg_info['img_name']}_obj{obj_counter}"
                        patch_registry.append({
                            "obj_id": obj_id,
                            "patch": cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                        })
                        clean_bbox = [class_id, float(x_c), float(y_c), float(bw), float(bh)]
                        obj_metadata[obj_id] = {
                            "area": area,
                            "grad_D": grad_D,
                            "surface_normal": surface_normal,
                            "c_sh": c_sh,
                            "solidity": solidity,
                            "source_image": img_path,
                            "bbox": clean_bbox,
                            "cluster": reg_info["cluster"],
                        }
                        obj_counter += 1

    # --- Phase B: DINOv2 Object Profiling ---
    logger.info("Phase B: DINOv2 object profiling (%d patches)...", len(patch_registry))
    patch_loader = DataLoader(
        PatchDataset(patch_registry, dino_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    embeddings_dict = {}

    with torch.no_grad():
        for batch_tensors, batch_ids in tqdm(patch_loader, desc="DINOv2 Object Profiling"):
            batch_tensors = batch_tensors.to(device)
            feats = dino(batch_tensors).cpu().numpy()
            for i, oid in enumerate(batch_ids):
                embeddings_dict[oid] = feats[i]

    # --- Isolation Forest & JSON Assembly ---
    logger.info("Applying Isolation Sieve per cluster...")
    latent_dictionary = {}
    clusters = sorted(list(set([v["cluster"] for v in obj_metadata.values()])))

    for c in tqdm(clusters, desc="Applying Isolation Sieve"):
        c_objs = {k: v for k, v in obj_metadata.items() if v.get("cluster") == c}
        obj_ids = list(c_objs.keys())

        if len(obj_ids) > 10:
            c_embeds = np.vstack([embeddings_dict[oid] for oid in obj_ids])
            iso_forest = IsolationForest(contamination=contamination, random_state=seed)
            iso_labels = iso_forest.fit_predict(c_embeds)
            for idx, lbl in enumerate(iso_labels):
                c_objs[obj_ids[idx]]["semantic_loss"] = bool(lbl == -1)
        else:
            for oid in obj_ids:
                c_objs[oid]["semantic_loss"] = False

        areas = [v["area"] for v in c_objs.values()]
        a_min = float(np.percentile(areas, 5)) if areas else 0.0
        a_max = float(np.percentile(areas, 95)) if areas else 1.0

        final_c_objs = {}
        for k, v in c_objs.items():
            v_copy = v.copy()
            v_copy.pop("cluster", None)
            final_c_objs[k] = v_copy

        latent_dictionary[c] = {
            "scale_bounds": {"A_min": a_min, "A_max": a_max},
            "objects": final_c_objs,
        }

    with open(output_path, "w") as f:
        json.dump(latent_dictionary, f, indent=4)

    # GPU cleanup
    del midas, dino, env_loader, patch_loader
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Dictionary compiled: %d objects → %s", obj_counter, output_path)
    return latent_dictionary
