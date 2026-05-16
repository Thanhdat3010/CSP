#!/usr/bin/env python3
"""
CHARM: Cluster-Aware Hard Example Mining Augmentation
For Camouflaged Object Detection (COD)

Method:
1. For each image in cluster C:
   - Find K hard negative images in same cluster (visual similarity)
   - Extract object from original image (using bbox)
   - Extract background from hard negative
   - Blend: object (original) + background (hard negative)
2. Result: Same object, but in challenging background context

Author: CSP Research
"""

import cv2
import numpy as np
import os
import shutil
from glob import glob
from tqdm import tqdm
import argparse
from pathlib import Path


def compute_image_descriptor(img):
    """
    Compute visual descriptor for image (for hard negative selection).
    Uses: Histogram + HOG-like gradient magnitude.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Color histogram (8 bins per channel)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_hist = cv2.calcHist([mag.astype(np.uint8)], [0], None, [8], [0, 256])
    mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()
    
    # Combine histograms
    descriptor = np.concatenate([hist, mag_hist])
    return descriptor


def descriptor_distance(desc1, desc2):
    """L2 distance between descriptors."""
    return np.linalg.norm(desc1 - desc2)


def yolo_to_bbox(label_line, h, w):
    """
    Convert YOLO format to pixel coordinates.
    YOLO format: [class_id, x_center_norm, y_center_norm, width_norm, height_norm]
    Returns: [x1, y1, x2, y2] in pixel coordinates
    """
    parts = label_line.strip().split()
    if len(parts) < 5:
        return None
    
    _, x_norm, y_norm, w_norm, h_norm = map(float, parts[:5])
    
    x_center = x_norm * w
    y_center = y_norm * h
    width = w_norm * w
    height = h_norm * h
    
    x1 = max(0, int(x_center - width / 2))
    y1 = max(0, int(y_center - height / 2))
    x2 = min(w, int(x_center + width / 2))
    y2 = min(h, int(y_center + height / 2))
    
    return [x1, y1, x2, y2]


def create_mask(img_h, img_w, bbox, margin=10):
    """
    Create binary mask for object region.
    mask = 1 (object, keep original)
    mask = 0 (background, take from hard negative)
    
    Args:
        img_h, img_w: Image dimensions
        bbox: [x1, y1, x2, y2] in pixels
        margin: Safety margin around object (pixels)
    
    Returns:
        mask: binary mask (uint8, 0-255)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    if bbox is None:
        return mask
    
    x1, y1, x2, y2 = bbox
    
    # Add margin (with clipping)
    x1_safe = max(0, x1 - margin)
    y1_safe = max(0, y1 - margin)
    x2_safe = min(img_w, x2 + margin)
    y2_safe = min(img_h, y2 + margin)
    
    mask[y1_safe:y2_safe, x1_safe:x2_safe] = 1
    
    return mask


def smooth_mask_boundary(mask, kernel_size=15):
    """
    Smooth mask boundary using Gaussian blur.
    This creates a soft transition at the object boundary.
    """
    # Dilate mask slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_smooth = cv2.dilate(mask.astype(np.float32), kernel, iterations=1)
    
    # Gaussian blur for soft transition
    mask_smooth = cv2.GaussianBlur(mask_smooth, (kernel_size, kernel_size), 0)
    mask_smooth = np.clip(mask_smooth, 0, 1)
    
    return mask_smooth


def blend_charm(img_original, img_hard_neg, bbox, margin=10, smooth_kernel=15):
    """
    Blend CHARM augmented image.
    
    Strategy:
    1. Extract object from original image (within bbox + margin)
    2. Extract background from hard negative image (outside bbox + margin)
    3. Blend with smooth boundary
    
    Args:
        img_original: Original image (easy detection)
        img_hard_neg: Hard negative image (same cluster, harder detection)
        bbox: Object bounding box [x1, y1, x2, y2]
        margin: Safety margin around object
        smooth_kernel: Kernel size for boundary smoothing
    
    Returns:
        augmented: Blended image
    """
    h, w = img_original.shape[:2]
    
    # Create mask
    mask = create_mask(h, w, bbox, margin=margin)
    
    # Smooth boundary
    mask_smooth = smooth_mask_boundary(mask, kernel_size=smooth_kernel)
    mask_smooth = np.stack([mask_smooth] * 3, axis=2)  # (H, W, 3)
    
    # Blend
    augmented = img_original * mask_smooth + img_hard_neg * (1 - mask_smooth)
    augmented = np.clip(augmented, 0, 255).astype(np.uint8)
    
    return augmented


def find_hard_negatives(cluster_images, cluster_labels, cluster_dir, top_k=3):
    """
    Find hard negative images within a cluster.
    
    Hard negative = image with same object but lower detection confidence.
    We use visual similarity as proxy: most similar images are usually hardest.
    
    Args:
        cluster_images: List of image filenames in cluster
        cluster_labels: List of label filenames in cluster
        cluster_dir: Path to cluster directory
        top_k: Number of hard negatives to return
    
    Returns:
        hard_negatives: List of (img_path, label_path) for hard negatives
    """
    if len(cluster_images) < top_k + 1:
        # Not enough images, return all except one
        return [(cluster_images[i], cluster_labels[i]) 
                for i in range(1, len(cluster_images))]
    
    # Compute descriptors for all images
    descriptors = []
    valid_indices = []
    
    for idx, img_name in enumerate(cluster_images):
        img_path = os.path.join(cluster_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        desc = compute_image_descriptor(img)
        descriptors.append(desc)
        valid_indices.append(idx)
    
    if len(descriptors) < 2:
        return []
    
    descriptors = np.array(descriptors)
    
    # For each image, find K most similar
    # (Assumption: most similar = most similar camouflage pattern = harder negative)
    hard_negatives_list = []
    
    for i in range(len(descriptors)):
        distances = [descriptor_distance(descriptors[i], descriptors[j]) 
                     for j in range(len(descriptors)) if i != j]
        
        # Get top-k closest (excluding self)
        sorted_indices = np.argsort(distances)[:top_k]
        
        for sorted_idx in sorted_indices:
            # Map back to original index
            if sorted_idx >= i:
                orig_idx = valid_indices[sorted_idx + 1]
            else:
                orig_idx = valid_indices[sorted_idx]
            
            img_name = cluster_images[orig_idx]
            label_name = cluster_labels[orig_idx]
            hard_negatives_list.append((img_name, label_name))
        
        break  # For efficiency, only process first image
    
    return hard_negatives_list


def generate_charm_dataset(partition_dir, output_dir, margin=10, smooth_kernel=15, 
                           top_k_hard=3, split_ratio=0.8):
    """
    Generate CHARM augmented dataset.
    
    Args:
        partition_dir: Path to COD10K_Partitioning directory
        output_dir: Output directory for augmented dataset
        margin: Safety margin around objects (pixels)
        smooth_kernel: Kernel size for boundary smoothing
        top_k_hard: Number of hard negatives per image
        split_ratio: Train/val split ratio (default: 80/20)
    """
    
    # Find all clusters
    train_img_dir = os.path.join(partition_dir, 'train', 'images')
    clusters = sorted([d for d in os.listdir(train_img_dir) 
                      if os.path.isdir(os.path.join(train_img_dir, d))])
    
    print(f"Found {len(clusters)} clusters")
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Process each cluster
    total_original = 0
    total_augmented = 0
    
    for cluster_name in tqdm(clusters, desc="Processing clusters"):
        cluster_img_dir = os.path.join(partition_dir, 'train', 'images', cluster_name)
        cluster_label_dir = os.path.join(partition_dir, 'train', 'labels', cluster_name)
        
        # Get all images and labels in cluster
        img_files = sorted([f for f in os.listdir(cluster_img_dir) if f.endswith('.jpg')])
        label_files = sorted([f.replace('.jpg', '.txt') for f in img_files])
        
        # For each image, generate CHARM augmented version
        for idx, img_name in enumerate(tqdm(img_files, desc=f"  {cluster_name}", leave=False)):
            label_name = label_files[idx]
            img_path = os.path.join(cluster_img_dir, img_name)
            label_path = os.path.join(cluster_label_dir, label_name)
            
            img = cv2.imread(img_path)
            if img is None or not os.path.exists(label_path):
                continue
            
            h, w = img.shape[:2]
            
            # Read label to get bbox
            with open(label_path, 'r') as f:
                label_line = f.readline()
            bbox = yolo_to_bbox(label_line, h, w)
            
            # Determine split
            split = 'train' if np.random.random() < split_ratio else 'val'
            
            # 1. Copy original image
            output_img_path = os.path.join(output_dir, split, 'images', img_name)
            output_label_path = os.path.join(output_dir, split, 'labels', label_name)
            
            shutil.copy(img_path, output_img_path)
            shutil.copy(label_path, output_label_path)
            total_original += 1
            
            # 2. Find hard negatives and generate augmented versions
            hard_negatives = find_hard_negatives(img_files, label_files, cluster_img_dir, 
                                               top_k=top_k_hard)
            
            for hn_idx, (hn_img_name, hn_label_name) in enumerate(hard_negatives):
                hn_img_path = os.path.join(cluster_img_dir, hn_img_name)
                hn_img = cv2.imread(hn_img_path)
                
                if hn_img is None:
                    continue
                
                # Resize hard negative to match original dimensions if needed
                if hn_img.shape[:2] != (h, w):
                    hn_img = cv2.resize(hn_img, (w, h))
                
                # Generate CHARM augmented image
                charm_img = blend_charm(img, hn_img, bbox, margin=margin, 
                                       smooth_kernel=smooth_kernel)
                
                # Save augmented image
                charm_img_name = os.path.splitext(img_name)[0] + f"_charm_hn{hn_idx}.jpg"
                charm_label_name = os.path.splitext(label_name)[0] + f"_charm_hn{hn_idx}.txt"
                
                charm_output_img = os.path.join(output_dir, split, 'images', charm_img_name)
                charm_output_label = os.path.join(output_dir, split, 'labels', charm_label_name)
                
                cv2.imwrite(charm_output_img, charm_img)
                shutil.copy(label_path, charm_output_label)
                total_augmented += 1
    
    print(f"\n✅ CHARM Dataset Generated!")
    print(f"   Total original images: {total_original}")
    print(f"   Total augmented images: {total_augmented}")
    print(f"   Output directory: {output_dir}")


def create_data_yaml(output_dir, num_classes=69):
    """Create data.yaml for YOLO training."""
    data_yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

nc: {num_classes}
names: [{{", ".join([f"class_{i}" for i in range(num_classes)])}}]
"""
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"Created {yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate CHARM augmented dataset for COD'
    )
    parser.add_argument('--partition_dir', type=str, 
                       default='data/COD10K_Partitioning',
                       help='Path to COD10K_Partitioning directory')
    parser.add_argument('--output', type=str, 
                       default='data/COD10K-CHARM',
                       help='Output directory for augmented dataset')
    parser.add_argument('--margin', type=int, default=10,
                       help='Safety margin around object (pixels)')
    parser.add_argument('--smooth_kernel', type=int, default=15,
                       help='Kernel size for boundary smoothing')
    parser.add_argument('--top_k_hard', type=int, default=3,
                       help='Number of hard negatives per image')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/val split ratio')
    parser.add_argument('--num_classes', type=int, default=69,
                       help='Number of object classes')
    
    args = parser.parse_args()
    
    # Generate dataset
    generate_charm_dataset(
        partition_dir=args.partition_dir,
        output_dir=args.output,
        margin=args.margin,
        smooth_kernel=args.smooth_kernel,
        top_k_hard=args.top_k_hard,
        split_ratio=args.split_ratio
    )
    
    # Create data.yaml
    create_data_yaml(args.output, num_classes=args.num_classes)
    
    print("\n🚀 Ready to train!")
    print(f"Command: yolo detect train model=yolov8s.pt data={args.output}/data.yaml ...")
