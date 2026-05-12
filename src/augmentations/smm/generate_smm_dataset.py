import cv2
import numpy as np
import os
import pickle
import shutil
from glob import glob
from tqdm import tqdm

def apply_smm_morphing(img, source_centroid, target_centroid, alpha=0.3, beta=0.1):
    # --- LAB Color Morphing ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Shift color distribution towards target centroid
    color_shift = alpha * (target_centroid['lab_mean'] - source_centroid['lab_mean'])
    lab += color_shift
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    morphed_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- FFT Frequency Morphing (FDA) ---
    # We apply FDA to the morphed BGR image
    h, w, c = morphed_bgr.shape
    new_img = np.zeros_like(morphed_bgr)
    
    for i in range(c):
        channel = morphed_bgr[:, :, i]
        fft = np.fft.fft2(channel)
        fft_shift = np.fft.fftshift(fft)
        
        amp = np.abs(fft_shift)
        phase = np.angle(fft_shift)
        
        # Target amplitude spectrum (resized to current image size)
        target_amp = cv2.resize(target_centroid['fft_amp_mean'], (w, h))
        
        # Replace low-frequency center
        cy, cx = h // 2, w // 2
        rh, rw = int(h * beta), int(w * beta)
        
        # Morph amplitude spectrum
        amp[cy-rh:cy+rh, cx-rw:cx+rw] = (1 - alpha) * amp[cy-rh:cy+rh, cx-rw:cx+rw] + alpha * target_amp[cy-rh:cy+rh, cx-rw:cx+rw]
        
        # Reconstruct
        fft_new = amp * np.exp(1j * phase)
        fft_ishift = np.fft.ifftshift(fft_new)
        img_back = np.fft.ifft2(fft_ishift)
        new_img[:, :, i] = np.clip(np.abs(img_back), 0, 255)

    return new_img

def generate_dataset(partition_dir, base_dataset_dir, centroids_file, output_dir):
    with open(centroids_file, 'rb') as f:
        centroids = pickle.load(f)
    
    cluster_names = list(centroids.keys())
    
    # Calculate mapping: Cluster -> Nearest Neighbor
    neighbor_map = {}
    for c1 in cluster_names:
        min_dist = float('inf')
        nearest = None
        for c2 in cluster_names:
            if c1 == c2: continue
            dist = np.linalg.norm(centroids[c1]['lab_mean'] - centroids[c2]['lab_mean'])
            if dist < min_dist:
                min_dist = dist
                nearest = c2
        neighbor_map[c1] = nearest

    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # Process Train (Original + Morphed)
    print("Generating SMM Augmented Training Set...")
    for cluster_name in cluster_names:
        img_paths = glob(os.path.join(partition_dir, 'train', 'images', cluster_name, "*.jpg"))
        target_cluster = neighbor_map[cluster_name]
        
        for img_path in tqdm(img_paths, desc=f"Processing {cluster_name}"):
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(partition_dir, 'train', 'labels', cluster_name, label_name)
            
            # 1. Copy Original
            shutil.copy(img_path, os.path.join(output_dir, 'train', 'images', img_name))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_dir, 'train', 'labels', label_name))
            
            # 2. Generate and Save Morphed
            img = cv2.imread(img_path)
            if img is None: continue
            
            morphed = apply_smm_morphing(img, centroids[cluster_name], centroids[target_cluster])
            
            morphed_name = os.path.splitext(img_name)[0] + "_smm.jpg"
            morphed_label_name = os.path.splitext(img_name)[0] + "_smm.txt"
            
            cv2.imwrite(os.path.join(output_dir, 'train', 'images', morphed_name), morphed)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_dir, 'train', 'labels', morphed_label_name))

    # Process Val (Copy As-Is)
    print("Copying Validation Set...")
    val_imgs = glob(os.path.join(base_dataset_dir, 'val', 'images', "*.jpg"))
    for img_path in tqdm(val_imgs):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(base_dataset_dir, 'val', 'labels', label_name)
        
        shutil.copy(img_path, os.path.join(output_dir, 'val', 'images', img_name))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_dir, 'val', 'labels', label_name))

if __name__ == "__main__":
    PARTITION_DIR = r"d:\Code\CSP\data\COD10K_Partitioning"
    BASE_DIR = r"d:\Code\CSP\data\COD10K-datasets"
    CENTROIDS = r"d:\Code\CSP\data\centroids.pkl"
    OUTPUT_DATASET = r"d:\Code\CSP\data\COD10K-SMM"
    
    generate_dataset(PARTITION_DIR, BASE_DIR, CENTROIDS, OUTPUT_DATASET)
    print(f"\nSMM Dataset Generation Complete: {OUTPUT_DATASET}")
