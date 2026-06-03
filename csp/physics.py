# -*- coding: utf-8 -*-
"""
CSP Physics & Math Functions.

All physically-grounded feature extraction functions used across the pipeline:
- Topographical features (surface normals from depth maps)
- Spherical Harmonics lighting approximation
- Adaptive solidity from skeleton analysis
- Local Binary Pattern (LBP) texture distance
- Z-buffer masking for depth-aware occlusion
- LAB color transfer for ambient tone matching
- Laplacian variance for blur detection
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.feature import local_binary_pattern

from . import config


# ==============================================================================
# Phase 1B — Dictionary Construction Features
# ==============================================================================

def get_topographical_features(depth_map, bbox, w, h):
    """Extract surface gradient and normal from a depth map region.

    Args:
        depth_map: Normalized depth map (H, W).
        bbox: YOLO-format bounding box [class_id, x_c, y_c, bw, bh].
        w, h: Image width and height.

    Returns:
        Tuple of (grad_D, surface_normal) where grad_D is [mean_gx, mean_gy]
        and surface_normal is the unit normal [nx, ny, nz].
    """
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2 + 1.0)
    norm_x = -grad_x / magnitude
    norm_y = -grad_y / magnitude
    norm_z = 1.0 / magnitude

    _, x_c, y_c, bw, bh = bbox
    x1 = max(0, int((x_c - bw / 2) * w))
    x2 = min(w, int((x_c + bw / 2) * w))
    y1 = max(0, int((y_c - bh / 2) * h))
    y2 = min(h, int((y_c + bh / 2) * h))

    region_gx = grad_x[y1:y2, x1:x2]
    region_gy = grad_y[y1:y2, x1:x2]
    region_nx = norm_x[y1:y2, x1:x2]
    region_ny = norm_y[y1:y2, x1:x2]
    region_nz = norm_z[y1:y2, x1:x2]

    if region_gx.size == 0:
        return [0.0, 0.0], [0.0, 0.0, 1.0]

    grad_D = [float(np.mean(region_gx)), float(np.mean(region_gy))]
    raw_mean_normal = [
        float(np.mean(region_nx)),
        float(np.mean(region_ny)),
        float(np.mean(region_nz)),
    ]

    n_mag = np.sqrt(sum(n**2 for n in raw_mean_normal)) + 1e-8
    surface_normal = [n / n_mag for n in raw_mean_normal]

    return grad_D, surface_normal


def get_spherical_harmonics(gray_img, bbox, w, h, mask=None):
    """Approximate first 9 Spherical Harmonics coefficients.

    Uses biological mask pixels when available to restrict the computation
    to the actual object region.

    Args:
        gray_img: Grayscale image normalized to [0, 1].
        bbox: YOLO-format bounding box [class_id, x_c, y_c, bw, bh].
        w, h: Image width and height.
        mask: Optional binary mask (255 = object).

    Returns:
        List of 9 SH coefficients.
    """
    _, x_c, y_c, bw, bh = bbox
    x1 = max(0, int((x_c - (bw * 1.5) / 2) * w))
    x2 = min(w, int((x_c + (bw * 1.5) / 2) * w))
    y1 = max(0, int((y_c - (bh * 1.5) / 2) * h))
    y2 = min(h, int((y_c + (bh * 1.5) / 2) * h))
    patch = gray_img[y1:y2, x1:x2].copy()

    if patch.size == 0:
        return [0.0] * 9

    if mask is not None:
        patch_mask = mask[y1:y2, x1:x2]
        if np.count_nonzero(patch_mask) > 0:
            patch = patch * (patch_mask.astype(np.float32) / 255.0)

    ph, pw = patch.shape
    y_grid, x_grid = np.mgrid[-1:1:ph * 1j, -1:1:pw * 1j]
    r2 = x_grid**2 + y_grid**2
    valid = r2 <= 1.0
    z_grid = np.zeros_like(x_grid)
    z_grid[valid] = np.sqrt(1 - r2[valid])

    Y = [
        np.ones_like(x_grid) * 0.282095,
        y_grid * 0.488603,
        z_grid * 0.488603,
        x_grid * 0.488603,
        (x_grid * y_grid) * 1.092548,
        (y_grid * z_grid) * 1.092548,
        (3.0 * z_grid**2 - 1.0) * 0.315392,
        (x_grid * z_grid) * 1.092548,
        (x_grid**2 - y_grid**2) * 0.546274,
    ]
    sum_valid = np.sum(valid)
    if sum_valid == 0:
        return [0.0] * 9

    return [float(np.sum(patch[valid] * basis[valid]) / sum_valid) for basis in Y]


def get_adaptive_solidity(mask):
    """Compute adaptive solidity using convex hull and skeleton thickness.

    For thin elongated objects (e.g., stick insects), the solidity is
    boosted to prevent over-penalizing valid camouflage strategies.

    Args:
        mask: Binary mask of the object region (uint8, 255 = foreground).

    Returns:
        Adaptive solidity score in [0, 1].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.50

    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    if area < 50:
        return 0.50

    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.50

    skeleton = skeletonize(mask > 0)
    skeleton_length = np.count_nonzero(skeleton)
    average_thickness = area / skeleton_length if skeleton_length > 0 else 10.0

    if average_thickness < 5.0:
        solidity = min(1.0, solidity + (5.0 - average_thickness) * 0.1)

    return float(solidity)


# ==============================================================================
# Phase 2 — Synthesis Validation & Harmonization
# ==============================================================================

def calculate_laplacian_variance(img_patch):
    """Measure image sharpness via Laplacian variance.

    Args:
        img_patch: BGR image patch.

    Returns:
        Variance of the Laplacian (higher = sharper).
    """
    gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_patch_surface_normal(depth_patch):
    """Compute mean surface normal from a depth patch.

    Args:
        depth_patch: Normalized depth patch (H, W).

    Returns:
        Unit surface normal [nx, ny, nz].
    """
    grad_x = cv2.Sobel(depth_patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_patch, cv2.CV_64F, 0, 1, ksize=3)
    mean_gx, mean_gy = np.mean(grad_x), np.mean(grad_y)
    magnitude = np.sqrt(mean_gx**2 + mean_gy**2 + 1.0)
    raw_normal = [-mean_gx / magnitude, -mean_gy / magnitude, 1.0 / magnitude]
    n_mag = np.sqrt(sum(n**2 for n in raw_normal)) + 1e-8
    return [n / n_mag for n in raw_normal]


def get_patch_spherical_harmonics(img_patch, mask=None):
    """Compute SH coefficients for a BGR image patch.

    Args:
        img_patch: BGR image patch.
        mask: Optional binary mask.

    Returns:
        List of 9 SH coefficients.
    """
    gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if mask is not None and np.count_nonzero(mask) > 0:
        gray = gray * (mask.astype(np.float32) / 255.0)

    ph, pw = gray.shape
    y_grid, x_grid = np.mgrid[-1:1:ph * 1j, -1:1:pw * 1j]
    r2 = x_grid**2 + y_grid**2
    valid = r2 <= 1.0
    z_grid = np.zeros_like(x_grid)
    z_grid[valid] = np.sqrt(1 - r2[valid])

    Y = [
        np.ones_like(x_grid) * 0.282095,
        y_grid * 0.488603,
        z_grid * 0.488603,
        x_grid * 0.488603,
        (x_grid * y_grid) * 1.092548,
        (y_grid * z_grid) * 1.092548,
        (3.0 * z_grid**2 - 1.0) * 0.315392,
        (x_grid * z_grid) * 1.092548,
        (x_grid**2 - y_grid**2) * 0.546274,
    ]
    sum_valid = np.sum(valid)
    if sum_valid == 0:
        return [0.0] * 9

    return [float(np.sum(gray[valid] * basis[valid]) / sum_valid) for basis in Y]


def calculate_lbp_distance(patch_a, patch_b):
    """Compute LBP histogram distance between two BGR patches.

    Args:
        patch_a: First BGR image patch.
        patch_b: Second BGR image patch.

    Returns:
        L2 distance between normalized LBP histograms.
    """
    gray_a = cv2.cvtColor(patch_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(patch_b, cv2.COLOR_BGR2GRAY)

    lbp_a = local_binary_pattern(gray_a, config.LBP_POINTS, config.LBP_RADIUS, method="uniform")
    lbp_b = local_binary_pattern(gray_b, config.LBP_POINTS, config.LBP_RADIUS, method="uniform")

    n_bins = config.LBP_POINTS + 2
    hist_a, _ = np.histogram(lbp_a.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist_b, _ = np.histogram(lbp_b.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))

    sum_a = hist_a.sum()
    sum_b = hist_b.sum()
    hist_a = hist_a.astype("float") / (sum_a if sum_a > 0 else 1.0)
    hist_b = hist_b.astype("float") / (sum_b if sum_b > 0 else 1.0)

    return np.linalg.norm(hist_a - hist_b)


def apply_z_buffer_masking(obj_mask, bg_depth_patch, obj_base_depth):
    """Occlude object pixels behind foreground depth elements.

    Args:
        obj_mask: Binary mask of the object.
        bg_depth_patch: Depth patch of the background region.
        obj_base_depth: Reference depth value for the object.

    Returns:
        Updated object mask with occluded pixels removed.
    """
    occlusion_mask = (
        bg_depth_patch > (obj_base_depth + config.Z_BUFFER_MARGIN)
    ).astype(np.uint8) * 255
    return cv2.bitwise_and(obj_mask, cv2.bitwise_not(occlusion_mask))


def transfer_color(source_patch, target_patch, mask):
    """Match ambient color tones via LAB color space transfer.

    Args:
        source_patch: Object BGR patch to adjust.
        target_patch: Background BGR patch to match against.
        mask: Binary mask defining object pixels.

    Returns:
        Tuple of (color-matched patch, shift magnitude).
    """
    src_lab = cv2.cvtColor(source_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask_bool = mask > 127

    if not np.any(mask_bool):
        return source_patch, 0.0

    src_mean, src_std = cv2.meanStdDev(src_lab, mask=mask)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

    shift_magnitude = np.sqrt(
        sum((src_mean[i][0] - tgt_mean[i][0]) ** 2 for i in range(3))
    )
    src_std[src_std == 0] = 1e-5

    result_lab = np.empty_like(src_lab)
    result_lab[:, :, 0] = (src_lab[:, :, 0] - src_mean[0][0]) + tgt_mean[0][0]

    for i in range(1, 3):
        std_ratio = np.clip(tgt_std[i][0] / src_std[i][0], 0.5, 1.5)
        result_lab[:, :, i] = (
            (src_lab[:, :, i] - src_mean[i][0]) * std_ratio
        ) + tgt_mean[i][0]

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    final_source = source_patch.copy()
    final_source[mask_bool] = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)[mask_bool]

    return final_source, shift_magnitude
