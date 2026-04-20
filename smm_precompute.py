import cv2
import numpy as np
import os
import pickle
from glob import glob
from tqdm import tqdm

def get_cluster_centroids(partition_dir):
    clusters = glob(os.path.join(partition_dir, "cluster_*"))
    centroids = {}

    print(f"Found {len(clusters)} clusters. Starting pre-computation...")

    for cluster_path in tqdm(clusters):
        cluster_name = os.path.basename(cluster_path)
        img_paths = glob(os.path.join(cluster_path, "*.jpg"))
        
        lab_sums = np.zeros(3, dtype=np.float64)
        fft_amp_sum = None
        count = 0

        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            
            # --- LAB Color Centroid ---
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab_sums += np.mean(lab, axis=(0, 1))
            
            # --- FFT Amplitude Centroid ---
            # Resize to a common size for FFT averaging (e.g., 256x256)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_res = cv2.resize(gray, (256, 256))
            fft = np.fft.fft2(gray_res)
            fft_shift = np.fft.fftshift(fft)
            amplitude = np.abs(fft_shift)
            
            if fft_amp_sum is None:
                fft_amp_sum = np.zeros_like(amplitude, dtype=np.float64)
            
            fft_amp_sum += amplitude
            count += 1

        if count > 0:
            centroids[cluster_name] = {
                'lab_mean': lab_sums / count,
                'fft_amp_mean': fft_amp_sum / count,
                'count': count
            }

    return centroids

if __name__ == "__main__":
    PARTITION_DIR = r"d:\Code\CSP\COD10K_Partitioning\train\images"
    OUTPUT_FILE = r"d:\Code\CSP\centroids.pkl"
    
    centroids = get_cluster_centroids(PARTITION_DIR)
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(centroids, f)
    
    print(f"\nPre-computation complete. Centroids saved to {OUTPUT_FILE}")
