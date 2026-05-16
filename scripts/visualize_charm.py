"""
Visualize CHARM Augmentation Method
Shows original image + 3 augmented variants side-by-side
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def get_images_in_cluster(cluster_dir):
    """Get all images directly inside a cluster directory."""
    if not os.path.exists(cluster_dir):
        return []
    return [f for f in os.listdir(cluster_dir) if f.endswith(('.jpg', '.png'))]


def compute_image_descriptor(img):
    """Compute 520-D descriptor for image similarity"""
    # Color histogram: 8 bins × 3 channels = 24 bins
    hist_b = cv2.calcHist([img], [0], None, [8], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [8], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [8], [0, 256])
    color_hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
    
    # Sobel gradients: magnitude in 8 bins
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    grad_hist = cv2.calcHist([magnitude.astype(np.uint8)], [0], None, [8], [0, 256])
    
    return np.concatenate([color_hist, grad_hist.flatten()])


def descriptor_distance(desc1, desc2):
    """L2 distance between descriptors"""
    return np.linalg.norm(desc1 - desc2)


def find_similar_images(cluster_images, cluster_dir, top_k=3):
    """Find K most similar images for hard negative mining"""
    if len(cluster_images) < 2:
        return cluster_images[1:] if len(cluster_images) > 1 else []
    
    # Compute descriptors
    descriptors = []
    valid_images = []
    
    for img_name in cluster_images:
        img_path = os.path.join(cluster_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        desc = compute_image_descriptor(img)
        descriptors.append(desc)
        valid_images.append(img_name)
    
    if len(descriptors) < 2:
        return []
    
    # Find most similar for first image
    distances = [descriptor_distance(descriptors[0], descriptors[j]) 
                 for j in range(1, len(descriptors))]
    top_indices = np.argsort(distances)[:top_k]
    
    return [valid_images[i+1] for i in top_indices]


def yolo_to_bbox(label_line, h, w):
    """Convert YOLO format to pixel bbox"""
    parts = label_line.strip().split()
    if len(parts) < 5:
        return None
    
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = max(0, int((cx - bw/2) * w))
    y1 = max(0, int((cy - bh/2) * h))
    x2 = min(w, int((cx + bw/2) * w))
    y2 = min(h, int((cy + bh/2) * h))
    
    return [x1, y1, x2, y2]


def create_mask(img_h, img_w, bbox, margin=10):
    """Create binary mask for object region"""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    if bbox is None:
        return mask
    
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img_w, x2 + margin)
    y2 = min(img_h, y2 + margin)
    
    mask[y1:y2, x1:x2] = 1
    return mask


def smooth_mask_boundary(mask, kernel_size=15):
    """Apply Gaussian blur to mask for smooth transitions"""
    mask_float = mask.astype(np.float32)
    mask_dilated = cv2.dilate(mask_float, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    mask_smooth = cv2.GaussianBlur(mask_dilated, (kernel_size, kernel_size), 0)
    return mask_smooth


def blend_charm(img_original, img_hard_neg, bbox, margin=10, smooth_kernel=15):
    """Blend original object with hard negative background"""
    h, w = img_original.shape[:2]
    
    # Create and smooth mask
    mask = create_mask(h, w, bbox, margin)
    mask_smooth = smooth_mask_boundary(mask, smooth_kernel)
    mask_smooth = mask_smooth / mask_smooth.max() if mask_smooth.max() > 0 else mask_smooth
    
    # Resize hard negative if needed
    if img_hard_neg.shape[:2] != (h, w):
        img_hard_neg = cv2.resize(img_hard_neg, (w, h))
    
    # Blend
    mask_3d = np.stack([mask_smooth] * 3, axis=2)
    augmented = (img_original * mask_3d + img_hard_neg * (1 - mask_3d)).astype(np.uint8)
    
    return augmented


def visualize_charm_method(dataset_dir, output_path='charm_visualization.jpg', num_samples=3):
    """
    Create visualization of CHARM method
    Shows: Original + 3 augmented variants for N samples
    """
    
    if os.path.isdir(os.path.join(dataset_dir, 'images')) and os.path.isdir(os.path.join(dataset_dir, 'labels')):
        partition_dir = os.path.join(dataset_dir, 'images')
        label_root = os.path.join(dataset_dir, 'labels')
    else:
        partition_dir = os.path.join(dataset_dir, 'train', 'images')
        label_root = os.path.join(dataset_dir, 'train', 'labels')
    
    if not os.path.exists(partition_dir):
        print(f"❌ Dataset directory not found: {partition_dir}")
        return
    
    # Get list of clusters
    clusters = [d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d))]
    
    if not clusters:
        print(f"❌ No clusters found in {partition_dir}")
        return
    
    print(f"📊 Found {len(clusters)} clusters")
    
    all_collages = []
    
    for sample_idx in range(min(num_samples, len(clusters))):
        cluster = clusters[sample_idx]
        cluster_dir = os.path.join(partition_dir, cluster)
        label_dir = os.path.join(label_root, cluster)
        
        cluster_images = get_images_in_cluster(cluster_dir)
        
        if not cluster_images:
            continue
        
        # Pick first image
        orig_img_name = cluster_images[0]
        orig_img_path = os.path.join(cluster_dir, orig_img_name)
        orig_img = cv2.imread(orig_img_path)
        
        if orig_img is None:
            continue
        
        h, w = orig_img.shape[:2]
        
        # Get label (bbox)
        label_name = Path(orig_img_name).with_suffix('.txt').name
        bbox = None
        if os.path.exists(os.path.join(label_dir, label_name)):
            with open(os.path.join(label_dir, label_name)) as f:
                label_line = f.readline()
                bbox = yolo_to_bbox(label_line, h, w)
        
        # Find hard negatives
        hard_neg_names = find_similar_images(cluster_images, cluster_dir, top_k=3)
        
        if not hard_neg_names:
            continue
        
        # Create variants
        variants = [orig_img]  # Original as first
        for hard_neg_name in hard_neg_names[:3]:
            hard_neg_path = os.path.join(cluster_dir, hard_neg_name)
            hard_neg_img = cv2.imread(hard_neg_path)
            
            if hard_neg_img is None:
                continue
            
            # Blend
            augmented = blend_charm(orig_img, hard_neg_img, bbox, margin=10, smooth_kernel=15)
            variants.append(augmented)
        
        if len(variants) < 4:
            continue
        
        # Resize all to same size for collage
        target_size = 320
        resized_variants = []
        for v in variants[:4]:
            resized = cv2.resize(v, (target_size, target_size))
            resized_variants.append(resized)
        
        # Create horizontal collage
        collage = np.hstack(resized_variants)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(collage, 'Original', (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(collage, f'Variant 1', (target_size + 10, 30), font, 0.8, (0, 165, 255), 2)
        cv2.putText(collage, f'Variant 2', (2*target_size + 10, 30), font, 0.8, (0, 165, 255), 2)
        cv2.putText(collage, f'Variant 3', (3*target_size + 10, 30), font, 0.8, (0, 165, 255), 2)
        
        # Add cluster name
        cv2.putText(collage, f'Cluster: {cluster}', (10, collage.shape[0] - 10), font, 0.7, (255, 255, 255), 2)
        
        all_collages.append(collage)
        print(f"✅ Sample {sample_idx + 1}: {cluster} (Original + 3 CHARM variants)")
    
    if not all_collages:
        print("❌ No valid samples to visualize")
        return
    
    # Stack all collages vertically
    final_collage = np.vstack(all_collages)
    
    # Save
    cv2.imwrite(output_path, final_collage)
    print(f"\n✅ Visualization saved: {output_path}")
    print(f"   Image size: {final_collage.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize CHARM Augmentation Method')
    parser.add_argument('--dataset', type=str, default='data/COD10K_Partitioning/train',
                        help='Path to dataset train folder')
    parser.add_argument('--output', type=str, default='charm_visualization.jpg',
                        help='Output visualization file')
    parser.add_argument('--samples', type=int, default=3,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    visualize_charm_method(args.dataset, args.output, args.samples)
