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
                        tasks.append({
                            "bg_path": bg_path,
                            "obj_data": obj,
                            "habitat": habitat,
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

    # Flatten tasks
    flattened_tasks = _flatten_tasks(environments, dictionary)
    logger.info("Flattened into %d atomic synthesis tasks.", len(flattened_tasks))

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

    logger.info("Starting synthesis loop...")
    with torch.no_grad():
        for batch_tensors, batch_paths, batch_obj_jsons, batch_valid, batch_h, batch_w in tqdm(
            dataloader, desc="Synthesizing Scenes"
        ):
            batch_tensors = batch_tensors.to(device)
            depth_preds = midas(batch_tensors)

            for i in range(len(batch_paths)):
                if not batch_valid[i]:
                    continue

                bg_path = batch_paths[i]
                obj = json.loads(batch_obj_jsons[i])
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
                possible_bg_masks = glob(
                    os.path.join(data_dir, "**", "mask", f"{bg_base_name}.*"), recursive=True
                )
                if possible_bg_masks:
                    loaded = cv2.imread(possible_bg_masks[0], cv2.IMREAD_GRAYSCALE)
                    if loaded is not None:
                        _, bg_mask = cv2.threshold(loaded, 127, 255, cv2.THRESH_BINARY)
                        if bg_mask.shape[:2] != (H, W):
                            bg_mask = cv2.resize(bg_mask, (W, H), interpolation=cv2.INTER_NEAREST)

                existing_bg_labels = []
                possible_bg_labels = glob(
                    os.path.join(data_dir, "**", "label", f"{bg_base_name}.txt"), recursive=True
                )
                if possible_bg_labels:
                    with open(possible_bg_labels[0], "r") as f:
                        existing_bg_labels = f.readlines()

                # --- Object Processing ---
                _, src_xc, src_yc, src_bw, src_bh = obj["bbox"]
                obj_base_name = os.path.splitext(os.path.basename(obj["source_image"]))[0]

                original_class_id = "-"
                possible_obj_labels = glob(
                    os.path.join(data_dir, "**", "label", f"{obj_base_name}.txt"), recursive=True
                )
                if possible_obj_labels:
                    with open(possible_obj_labels[0], "r") as f_label:
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

                possible_obj_masks = glob(
                    os.path.join(data_dir, "**", "mask", f"{obj_base_name}.*"), recursive=True
                )
                if not possible_obj_masks:
                    continue

                obj_src_mask = cv2.imread(possible_obj_masks[0], cv2.IMREAD_GRAYSCALE)
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

                # --- Placement Retry Loop ---
                for attempt in range(max_retries):
                    pad_x_pct = 0.05
                    pad_y_pct = 0.05
                    padded_bw = src_bw + (src_bw * pad_x_pct * 2)
                    padded_bh = src_bh + (src_bh * pad_y_pct * 2)

                    target_w = int(padded_bw * W)
                    target_h = int(padded_bh * H)
                    if target_w >= W or target_h >= H or target_w <= 10 or target_h <= 10:
                        break

                    prop_x = random.randint(5, max(5, W - target_w - 5))
                    prop_y = random.randint(5, max(5, H - target_h - 5))

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
