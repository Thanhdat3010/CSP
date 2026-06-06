# -*- coding: utf-8 -*-
"""
CSP Phase 2 — Master Synthesis Engine.

Corresponds to Cell 8 of the original notebook:
1. Task-Centric flattening of all (background, object) pairs
2. MiDaS batched depth for validation gates
3. Multi-gate validation: Scale, Topography, Lighting (SH), Texture (LBP)
4. Harmonization: Z-Buffer, LAB Color Transfer, Poisson Blending
5. Ground-truth generation (bbox + mask)
"""

import os
import gc
import json
import random
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .utils import seed_everything, get_device, ensure_dir
from .models import load_midas
from .datasets import AtomicSynthesisDataset
from .physics import (
    calculate_laplacian_variance,
    get_patch_surface_normal,
    get_patch_spherical_harmonics,
    calculate_lbp_distance,
    apply_z_buffer_masking,
    transfer_color,
)

logger = logging.getLogger("csp.synthesis")


def _flatten_tasks(environments: dict, dictionary: dict) -> list:
    """Flatten all valid (background, object) pairs into atomic tasks.

    Args:
        environments: Background catalog from ingestion phase.
        dictionary: Latent Affordance Dictionary.

    Returns:
        List of task dicts, each with 'bg_path', 'obj_data', 'habitat'.
    """
    tasks = []

    for env_name, env_data in environments.items():
        if isinstance(env_data, dict) and "path" in env_data:
            items = [(env_name, env_data)]
        else:
            items = env_data.items()

        for item_key, item_meta in items:
            habitat = (
                env_name
                if not isinstance(env_data, dict) or "path" not in env_data
                else env_data.get("matched_habitat", "global_habitat")
            )
            bg_path = item_meta["path"]

            if habitat in dictionary and dictionary[habitat]["objects"]:
                habitat_objects = list(dictionary[habitat]["objects"].values())
                scale = dictionary[habitat]["scale_bounds"]

                for obj in habitat_objects:
                    if scale["A_min"] <= obj["area"] <= scale["A_max"]:
                        # O(1) Pre-filtering Checks
                        bg_orig_w = item_meta["orig_w"]
                        bg_orig_h = item_meta["orig_h"]
                        r_avail_px = item_meta["R_avail_norm"] * bg_orig_w
                        
                        _, _, _, bw, bh = obj["bbox"]
                        obj_w_px = bw * bg_orig_w
                        obj_h_px = bh * bg_orig_h
                        obj_max_dim = max(obj_w_px, obj_h_px)
                        
                        # Spatial Capacity Check
                        if obj_max_dim > (r_avail_px * 2):
                            continue
                        
                        # Global Lighting Check
                        if "global_sh" in item_meta:
                            sh_bg = item_meta["global_sh"]
                            sh_obj = obj["c_sh"]
                            if np.sum(np.abs(sh_bg)) > 0 and np.sum(np.abs(sh_obj)) > 0:
                                sim = 1.0 - cosine(sh_obj, sh_bg)
                                if sim < 0.60:
                                    continue

                        bg_meta_copy = item_meta.copy()
                        bg_meta_copy["habitat"] = habitat

                        tasks.append({
                            "bg_path": bg_path,
                            "obj_data": obj,
                            "habitat": habitat,
                            "bg_meta": bg_meta_copy,
                        })

    return tasks


def synthesize(
    catalog_path: str,
    dictionary_path: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = config.SYNTHESIS_BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    max_retries: int = config.MAX_RETRIES,
    seed: int = config.SEED,
    max_synthesis_images: int = config.MAX_SYNTHESIS_IMAGES,
):
    """Execute the full synthesis engine.

    Args:
        catalog_path: Path to environment_catalog.json.
        dictionary_path: Path to latent_affordance_dictionary.json.
        data_dir: Root of original dataset (for masks/labels lookup).
        output_dir: Output directory for synthesized images.
        batch_size: Batch size for MiDaS.
        num_workers: DataLoader workers.
        max_retries: Max placement attempts per object.
        seed: Random seed.
        max_synthesis_images: Max number of successfully synthesized images to generate.

    Returns:
        Number of successfully synthesized images.
    """
    seed_everything(seed)
    random.seed(seed)
    device = get_device()
    ensure_dir(output_dir)

    # Load dictionaries
    with open(catalog_path, "r") as f:
        environments = json.load(f)
    with open(dictionary_path, "r") as f:
        dictionary = json.load(f)

    # Load MiDaS
    midas, midas_transforms = load_midas(device)

    # Pre-scan mask and label files in data_dir once for O(1) lookups
    logger.info("Pre-scanning dataset masks and labels in %s...", data_dir)
    mask_files = {}
    label_files = {}
    for root, dirs, files in os.walk(data_dir):
        path_parts = root.replace("\\", "/").split("/")
        if "mask" in path_parts:
            for f in files:
                if f.endswith((".png", ".jpg", ".jpeg")):
                    base = os.path.splitext(f)[0]
                    mask_files[base] = os.path.join(root, f)
        elif "label" in path_parts:
            for f in files:
                if f.endswith(".txt"):
                    base = os.path.splitext(f)[0]
                    label_files[base] = os.path.join(root, f)
    logger.info("Found %d masks and %d labels in dataset directory.", len(mask_files), len(label_files))

    # Flatten tasks
    flattened_tasks = _flatten_tasks(environments, dictionary)
    logger.info("Flattened into %d atomic synthesis tasks.", len(flattened_tasks))

    # Shuffle and limit candidate tasks to prevent massive dataloader overhead
    random.shuffle(flattened_tasks)
    max_candidates = max_synthesis_images * 3
    if len(flattened_tasks) > max_candidates:
        flattened_tasks = flattened_tasks[:max_candidates]
        logger.info("Limited synthesis candidates to %d tasks for performance.", len(flattened_tasks))

    # DataLoader
    dataloader = DataLoader(
        AtomicSynthesisDataset(flattened_tasks, midas_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    success_count = 0
    source_image_cache = {}
    dt_cache = {}

    logger.info("Starting synthesis loop (max limit: %d)...", max_synthesis_images)
    with torch.no_grad():
        for batch_tensors, batch_paths, batch_obj_jsons, batch_valid, batch_h, batch_w, batch_bg_meta_jsons in tqdm(
            dataloader, desc="Synthesizing Scenes"
        ):
            if success_count >= max_synthesis_images:
                logger.info("Reached max synthesis image limit (%d). Breaking loop.", max_synthesis_images)
                break

            batch_tensors = batch_tensors.to(device)
            depth_preds = midas(batch_tensors)

            for i in range(len(batch_paths)):
                if success_count >= max_synthesis_images:
                    break
                if not batch_valid[i]:
                    continue

                bg_path = batch_paths[i]
                obj = json.loads(batch_obj_jsons[i])
                bg_meta = json.loads(batch_bg_meta_jsons[i])
                H, W = batch_h[i].item(), batch_w[i].item()
                bg_base_name = os.path.splitext(os.path.basename(bg_path))[0]

                bg_depth = torch.nn.functional.interpolate(
                    depth_preds[i].unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
                bg_depth = (bg_depth - bg_depth.min()) / (bg_depth.max() - bg_depth.min() + 1e-8)

                bg_img = cv2.imread(bg_path)

                # Background mask & labels
                bg_mask = np.zeros((H, W), dtype=np.uint8)
                bg_mask_path = mask_files.get(bg_base_name)
                if bg_mask_path:
                    loaded = cv2.imread(bg_mask_path, cv2.IMREAD_GRAYSCALE)
                    if loaded is not None:
                        _, bg_mask = cv2.threshold(loaded, 127, 255, cv2.THRESH_BINARY)
                        if bg_mask.shape[:2] != (H, W):
                            bg_mask = cv2.resize(bg_mask, (W, H), interpolation=cv2.INTER_NEAREST)

                existing_bg_labels = []
                bg_label_path = label_files.get(bg_base_name)
                if bg_label_path:
                    with open(bg_label_path, "r") as f:
                        existing_bg_labels = f.readlines()

                # --- Object Processing ---
                _, src_xc, src_yc, src_bw, src_bh = obj["bbox"]
                obj_base_name = os.path.splitext(os.path.basename(obj["source_image"]))[0]

                original_class_id = "-"
                obj_label_path = label_files.get(obj_base_name)
                if obj_label_path:
                    with open(obj_label_path, "r") as f_label:
                        for line in f_label:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    lbl_xc, lbl_yc = float(parts[1]), float(parts[2])
                                    if abs(lbl_xc - src_xc) < 1e-3 and abs(lbl_yc - src_yc) < 1e-3:
                                        original_class_id = str(int(float(parts[0])))
                                        break
                                except ValueError:
                                    pass

                obj_mask_path = mask_files.get(obj_base_name)
                if not obj_mask_path:
                    continue

                obj_src_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)
                if obj_src_mask is None:
                    continue

                # Source image cache
                src_path = obj["source_image"]
                if src_path not in source_image_cache:
                    img_load = cv2.imread(src_path)
                    if img_load is None:
                        continue
                    source_image_cache[src_path] = img_load
                    if len(source_image_cache) > config.SOURCE_CACHE_SIZE:
                        source_image_cache.pop(next(iter(source_image_cache)))

                src_img = source_image_cache[src_path]
                src_H, src_W = src_img.shape[:2]

                # --- Deterministic Scale-Space Search ---
                # Load or compute DT map for this background
                if bg_path not in dt_cache:
                    empty_space = np.uint8((bg_mask == 0) * 255)
                    dt_map = cv2.distanceTransform(empty_space, cv2.DIST_L2, 5)
                    dt_cache[bg_path] = dt_map
                else:
                    dt_map = dt_cache[bg_path]

                # Load feature grids
                try:
                    bg_spatial_grid = np.load(bg_meta["feature_path"])
                    obj_spatial_grid = np.load(obj["feature_path"])
                except Exception as e:
                    logger.warning("Failed to load feature grids for task: %s", e)
                    continue

                # Scale Space bounds
                import math
                area_orig = obj["area"]
                habitat = bg_meta.get("habitat", "cluster_1")
                scale_bounds = dictionary.get(habitat, {}).get("scale_bounds", {"A_min": 0.01, "A_max": 0.20})
                A_min = scale_bounds["A_min"]
                A_max = scale_bounds["A_max"]

                min_scale = math.sqrt(max(1e-5, A_min) / max(1e-5, area_orig))
                max_scale = math.sqrt(A_max / max(1e-5, area_orig))

                min_scale = max(0.3, min_scale)
                max_scale = min(1.5, max_scale)
                if min_scale > max_scale:
                    min_scale, max_scale = max_scale, min_scale

                candidate_scales = np.linspace(min_scale, max_scale, num=5)
                obj_max_dim = max(src_bw * W, src_bh * H)
                base_obj_radius = obj_max_dim / 2.0

                # Convert to tensors
                bg_grid = torch.tensor(bg_spatial_grid).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H_bg, W_bg)
                obj_grid = torch.tensor(obj_spatial_grid).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H_obj, W_obj)
                dt_map_tensor = torch.tensor(dt_map).to(device)  # (H, W)

                best_score = -1.0
                best_params = (None, None, None)

                for scale in candidate_scales:
                    scaled_obj_grid = F.interpolate(
                        obj_grid,
                        scale_factor=scale,
                        mode="bilinear",
                        align_corners=False
                    )

                    obj_vector = scaled_obj_grid.mean(dim=(2, 3)).squeeze()  # (C,)
                    bg_vectors = bg_grid.squeeze().permute(1, 2, 0)  # (H_bg, W_bg, C)

                    # Cosine similarity heatmap
                    semantic_heatmap = F.cosine_similarity(
                        bg_vectors,
                        obj_vector.view(1, 1, -1),
                        dim=-1
                    )  # (H_bg, W_bg)

                    # Resize heatmap to match full image dimensions
                    semantic_heatmap_full = F.interpolate(
                        semantic_heatmap.unsqueeze(0).unsqueeze(0),
                        size=dt_map_tensor.shape,
                        mode="bilinear",
                        align_corners=False
                    ).squeeze()  # (H, W)

                    # Apply Spatial Constraint Mask
                    current_radius = base_obj_radius * scale
                    spatial_mask = (dt_map_tensor >= current_radius).float()

                    # Suitability Map
                    suitability = semantic_heatmap_full * spatial_mask

                    max_val = suitability.max().item()
                    if max_val > best_score and max_val > 0.0:
                        best_score = max_val
                        max_idx = (suitability == max_val).nonzero(as_tuple=True)
                        optimal_y, optimal_x = max_idx[0][0].item(), max_idx[1][0].item()
                        best_params = (scale, optimal_y, optimal_x)

                final_scale, optimal_y, optimal_x = best_params
                if final_scale is None:
                    continue  # Object cannot physically fit, discard task

                # Single-attempt loop to preserve downstream variable indentation
                for attempt in range(1):
                    pad_x_pct = 0.05
                    pad_y_pct = 0.05
                    padded_bw = (src_bw + (src_bw * pad_x_pct * 2)) * final_scale
                    padded_bh = (src_bh + (src_bh * pad_y_pct * 2)) * final_scale

                    target_w = int(padded_bw * W)
                    target_h = int(padded_bh * H)
                    if target_w >= W or target_h >= H or target_w <= 10 or target_h <= 10:
                        break

                    prop_x = int(optimal_x - target_w // 2)
                    prop_y = int(optimal_y - target_h // 2)

                    prop_x = max(5, min(W - target_w - 5, prop_x))
                    prop_y = max(5, min(H - target_h - 5, prop_y))

                    bg_patch = bg_img[prop_y : prop_y + target_h, prop_x : prop_x + target_w]
                    actual_h, actual_w = bg_patch.shape[:2]
                    if actual_h <= 5 or actual_w <= 5:
                        continue

                    bg_mask_patch = bg_mask[prop_y : prop_y + actual_h, prop_x : prop_x + actual_w]
                    if np.any(bg_mask_patch > 0):
                        continue

                    bg_depth_patch = bg_depth[prop_y : prop_y + actual_h, prop_x : prop_x + actual_w]

                    x_min = int((src_xc - src_bw / 2.0) * src_W)
                    x_max = int((src_xc + src_bw / 2.0) * src_W)
                    y_min = int((src_yc - src_bh / 2.0) * src_H)
                    y_max = int((src_yc + src_bh / 2.0) * src_H)

                    p_x = int(src_bw * pad_x_pct * src_W)
                    p_y = int(src_bh * pad_y_pct * src_H)

                    sx1 = max(0, x_min - p_x)
                    sx2 = min(src_W, x_max + p_x)
                    sy1 = max(0, y_min - p_y)
                    sy2 = min(src_H, y_max + p_y)

                    if (sx2 - sx1) < (x_max - x_min) or (sy2 - sy1) < (y_max - y_min):
                        break

                    raw_obj_crop = src_img[sy1:sy2, sx1:sx2]
                    if raw_obj_crop.size == 0:
                        break

                    # 50% horizontal flip
                    is_flipped = random.random() > 0.5
                    if is_flipped:
                        raw_obj_crop = cv2.flip(raw_obj_crop, 1)

                    if obj_src_mask.shape[:2] != (src_H, src_W):
                        resized_mask = cv2.resize(obj_src_mask, (src_W, src_H), interpolation=cv2.INTER_NEAREST)
                    else:
                        resized_mask = obj_src_mask

                    raw_mask_crop = resized_mask[sy1:sy2, sx1:sx2]
                    if is_flipped:
                        raw_mask_crop = cv2.flip(raw_mask_crop, 1)

                    obj_mask_resized = cv2.resize(raw_mask_crop, (actual_w, actual_h), interpolation=cv2.INTER_NEAREST)
                    _, obj_mask_resized = cv2.threshold(obj_mask_resized, 127, 255, cv2.THRESH_BINARY)

                    # Connected components cleanup
                    num_labels, labels_cc, stats, centroids_cc = cv2.connectedComponentsWithStats(
                        obj_mask_resized, connectivity=8
                    )
                    if num_labels > 2:
                        center_x, center_y = actual_w // 2, actual_h // 2
                        best_label = 1
                        min_dist = float("inf")
                        for l in range(1, num_labels):
                            cx, cy = centroids_cc[l]
                            dist = (cx - center_x) ** 2 + (cy - center_y) ** 2
                            if dist < min_dist:
                                min_dist = dist
                                best_label = l
                        obj_mask_resized = np.where(labels_cc == best_label, 255, 0).astype(np.uint8)

                    # Z-buffer masking for anomalous objects
                    if obj.get("semantic_loss", False):
                        obj_mask_resized = apply_z_buffer_masking(
                            obj_mask_resized, bg_depth_patch, np.median(bg_depth_patch)
                        )
                        if np.count_nonzero(obj_mask_resized) < (actual_w * actual_h * 0.2):
                            continue

                    # Validation gates
                    if calculate_laplacian_variance(bg_patch) < config.BLUR_THRESHOLD:
                        continue

                    if np.dot(obj["surface_normal"], get_patch_surface_normal(bg_depth_patch)) < config.SURFACE_NORMAL_THRESHOLD:
                        continue

                    sh_bg = get_patch_spherical_harmonics(bg_patch, mask=obj_mask_resized)
                    sh_obj = obj["c_sh"]

                    if np.sum(np.abs(sh_bg)) == 0 or np.sum(np.abs(sh_obj)) == 0:
                        continue

                    try:
                        if (1.0 - cosine(sh_obj, sh_bg)) < config.SH_SIMILARITY_THRESHOLD:
                            continue
                    except Exception:
                        continue

                    obj_pixels = cv2.resize(raw_obj_crop, (actual_w, actual_h))
                    color_matched_obj, shift_mag = transfer_color(obj_pixels, bg_patch, obj_mask_resized)
                    if shift_mag > config.COLOR_SHIFT_MAX:
                        continue

                    isolated = cv2.bitwise_and(obj_pixels, obj_pixels, mask=obj_mask_resized)
                    if calculate_lbp_distance(isolated, bg_patch) > (0.6 * obj.get("solidity", 0.50)):
                        continue

                    # --- Harmonization & Blending ---
                    pristine_obj_mask = obj_mask_resized.copy()
                    temp_bg = bg_img.copy()

                    erosion_kernel = np.ones((5, 5), np.uint8)
                    blend_mask = cv2.erode(obj_mask_resized, erosion_kernel, iterations=1)
                    _, hard_binary_mask = cv2.threshold(blend_mask, 127, 255, cv2.THRESH_BINARY)
                    if np.count_nonzero(hard_binary_mask) < 10:
                        continue

                    center_coord = (prop_x + actual_w // 2, prop_y + actual_h // 2)

                    try:
                        harmonized_canvas = cv2.seamlessClone(
                            color_matched_obj, temp_bg.copy(), hard_binary_mask, center_coord, cv2.NORMAL_CLONE
                        )
                        harmonized_patch = harmonized_canvas[prop_y : prop_y + actual_h, prop_x : prop_x + actual_w]
                    except Exception:
                        continue

                    alpha = cv2.GaussianBlur(pristine_obj_mask, (3, 3), 0).astype(np.float32) / 255.0
                    if len(alpha.shape) == 2:
                        alpha = np.expand_dims(alpha, axis=2)

                    bg_float = temp_bg[prop_y : prop_y + actual_h, prop_x : prop_x + actual_w].astype(np.float32)
                    harm_float = harmonized_patch.astype(np.float32)
                    final_patch = (harm_float * alpha) + (bg_float * (1.0 - alpha))
                    final_patch = np.clip(final_patch, 0, 255).astype(np.uint8)
                    temp_bg[prop_y : prop_y + actual_h, prop_x : prop_x + actual_w] = final_patch

                    # Ground truth generation
                    full_obj_mask = np.zeros((H, W), dtype=np.uint8)
                    full_obj_mask[prop_y : prop_y + actual_h, prop_x : prop_x + actual_w] = pristine_obj_mask
                    final_merged_mask = cv2.bitwise_or(bg_mask, full_obj_mask)

                    mask_contours, _ = cv2.findContours(pristine_obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if mask_contours:
                        all_pts = np.vstack(mask_contours)
                        x_b, y_b, w_b, h_b = cv2.boundingRect(all_pts)
                        tight_xc = prop_x + x_b + w_b / 2.0
                        tight_yc = prop_y + y_b + h_b / 2.0
                        tight_w = float(w_b)
                        tight_h = float(h_b)
                    else:
                        tight_xc = prop_x + actual_w / 2.0
                        tight_yc = prop_y + actual_h / 2.0
                        tight_w = float(actual_w)
                        tight_h = float(actual_h)

                    new_bbox = f"{original_class_id} {tight_xc / W:.6f} {tight_yc / H:.6f} {tight_w / W:.6f} {tight_h / H:.6f}\n"
                    final_labels = existing_bg_labels.copy()
                    final_labels.append(new_bbox)

                    # Write outputs
                    cv2.imwrite(os.path.join(output_dir, f"synth_{success_count}.jpg"), temp_bg)
                    cv2.imwrite(os.path.join(output_dir, f"synth_{success_count}.png"), final_merged_mask)
                    with open(os.path.join(output_dir, f"synth_{success_count}.txt"), "w") as f:
                        f.writelines(final_labels)

                    success_count += 1
                    break  # Exit retry loop

    # Cleanup
    del midas, dataloader
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Synthesis complete: %d images generated → %s", success_count, output_dir)
    return success_count
