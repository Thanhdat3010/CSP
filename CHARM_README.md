# CHARM - Cluster-Aware Hard Example Mining for COD

## 🎯 Overview

CHARM is a novel data augmentation method for Camouflaged Object Detection (COD) that:

1. **Preserves Objects** - Keeps object regions exactly as they are (100% unchanged)
2. **Finds Hard Negatives** - Identifies harder-to-detect images within the same cluster
3. **Blends Intelligently** - Replaces background with hard negative's background + smooth boundary

**Result:** Training images with same object but challenging context → Better recall without losing precision.

---

## 📊 Quick Start

### Step 1: Generate CHARM Dataset (Local - 15 min)
```bash
cd d:\Code\CSP
.\charm_quickstart.bat
```

**Expected Output:**
- Dataset location: `data/COD10K-CHARM/`
- Train images: 24,320 (6,080 original + 18,240 augmented)
- Val images: 2,026 (original only)
- Verification collage: `charm_verification_collage.jpg`

### Step 2: Upload & Train on Colab (GPU - 1.2 hours)

Upload `data/COD10K-CHARM/` to Google Drive, then in Colab:

```python
# Mount and train
from google.colab import drive
drive.mount('/content/drive')

!yolo detect train \
    model=yolov8s.pt \
    data=/content/drive/MyDrive/COD10K-CHARM/data.yaml \
    epochs=100 \
    batch=64 \
    optimizer=AdamW \
    lr0=0.0001 \
    project=/content/drive/MyDrive/runs/CHARM \
    name=charm_v1
```

**Output:**
- Best model: `/runs/CHARM/charm_v1/weights/best.pt`
- Metrics: `/runs/CHARM/charm_v1/results.csv`

---

## 🔧 Advanced Usage

### Custom Parameters

```bash
python src/augmentations/charm/generate_charm_dataset.py ^
    --partition_dir data/COD10K_Partitioning ^
    --output data/COD10K-CHARM ^
    --margin 10 ^
    --smooth_kernel 15 ^
    --top_k_hard 3 ^
    --split_ratio 0.8
```

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--margin` | 10 | 5-20 | Safety buffer around object (pixels) |
| `--smooth_kernel` | 15 | 5-21 | Gaussian blur kernel for boundary |
| `--top_k_hard` | 3 | 1-5 | Number of hard negatives per image |
| `--split_ratio` | 0.8 | 0.7-0.9 | Train/val split ratio |

### Ablation Studies

**Test different K values:**
```bash
python src/augmentations/charm/generate_charm_dataset.py --top_k_hard 1 --output data/COD10K-CHARM-K1
python src/augmentations/charm/generate_charm_dataset.py --top_k_hard 2 --output data/COD10K-CHARM-K2
python src/augmentations/charm/generate_charm_dataset.py --top_k_hard 5 --output data/COD10K-CHARM-K5
```

---

## 📁 File Structure

```
src/augmentations/charm/
├── generate_charm_dataset.py    # Main implementation (300+ lines)
│   ├── compute_image_descriptor()      # HOG + Histogram
│   ├── find_hard_negatives()          # Similarity-based selection
│   ├── yolo_to_bbox()                 # Format conversion
│   ├── create_mask()                  # Binary mask creation
│   ├── smooth_mask_boundary()         # Gaussian smoothing
│   ├── blend_charm()                  # Main blending function
│   └── generate_charm_dataset()       # Full pipeline
│
charm_generate.bat              # Quick generation script
charm_train.bat                 # Quick training script
cleanup_smm.bat                 # Remove legacy SMM files

data/COD10K-CHARM/
├── train/
│   ├── images/  (24,320 images)
│   └── labels/  (24,320 labels)
├── val/
│   ├── images/  (2,026 images)
│   └── labels/  (2,026 labels)
└── data.yaml    (YOLO config)
```

---

## 🧪 How CHARM Works (Visual)

```
Ảnh gốc (dễ)          Ảnh khó (same cluster)      CHARM Augmented
┌──────────────┐      ┌──────────────┐            ┌──────────────┐
│   ╔════╗     │      │   ╔════╗     │            │   ╔════╗     │
│   ║ CÁO║ clear│      │   ║ CÁO║ blended        │   ║ CÁO║ clear│
│   ╚════╝     │      │   ╚════╝ in hard        │   ╚════╝     │
│ [easy grass] │      │  [hard grass]           │ [hard grass]  │
└──────────────┘      └──────────────┘          └──────────────┘

Confidence:           Confidence:               Confidence:
- Model: 0.95        - Model: 0.30             - Model: 0.85
- Easy to detect     - Hard to detect          - Challenging but learnable
                                               
"Teaching model to detect object even in hard context"
```

---

## 📈 Expected Results

### Performance Metrics

| Metric | Baseline | CHARM (Expected) | Improvement |
|--------|----------|-----------------|-------------|
| Precision | 0.596 | 0.55-0.65 | ✅ Stable/Better |
| Recall | 0.148 | 0.20-0.25 | ⭐ +35-70% |
| mAP@50 | 0.166 | 0.18-0.22 | ⭐ +8-32% |
| mAP@50-95 | 0.107 | 0.12-0.14 | ⭐ +12-31% |

### Class-wise Improvement (Expected)

Hardest classes will improve most:
- Dog: 0% → 15-20% ⭐⭐⭐
- Wolf: 0% → 12-18% ⭐⭐⭐
- Fish: 14% → 20-25% ⭐⭐
- Frog: 8% → 15-20% ⭐⭐
- Chameleon: 7% → 22-28% ⭐⭐⭐

---

## 🔍 Core Algorithms

### Hard Negative Selection
```
1. For each image, compute descriptor:
   - Color histogram (8×8×8 bins in RGB)
   - Gradient magnitude histogram
   
2. Find K most similar images in same cluster
   (Assumption: similar appearance = similar camouflage = harder)
   
3. Select top-K as hard negatives
```

### Intelligent Blending
```
1. Create mask from object bbox + margin
   - Mask = 1 inside bbox + margin (object region)
   - Mask = 0 outside (background region)

2. Smooth mask boundary with Gaussian blur
   - Prevents hard edge artifacts
   - Smooth transition from object to background

3. Blend with hard negative:
   - augmented = object × mask + hard_neg × (1 - mask)
   
Result: Original object + Challenging background
```

---

## 📚 Why CHARM Works

| Aspect | Why | Example |
|--------|-----|---------|
| **Cluster-Aware** | Same camouflage type | "Fox in brown leaves" + "Fox in brown leaves" |
| **Hard Negatives** | Real examples, not synthetic | Actual hard-to-detect foxes, not generated |
| **Object Preservation** | Maintains confidence | Model still recognizes object clearly |
| **Smooth Boundary** | Realistic augmentation | No artifacts that model can exploit |

---

## ⚙️ Technical Details

### Descriptor Computation
- **Color Histogram**: 8 bins per RGB channel = 512 values
- **Gradient Magnitude**: Sobel filter + 8 bins = 8 values
- **Total**: 520-dimensional descriptor
- **Distance**: L2 norm between descriptors

### Mask Smoothing
- **Kernel Size**: 15×15 (default, customizable)
- **Method**: Gaussian blur on dilated mask
- **Range**: [0, 1] smooth transition (not binary)

### Blending
- **Formula**: `aug = orig × mask_smooth + hard_neg × (1 - mask_smooth)`
- **Range**: Clipped to [0, 255] uint8

---

## 🐛 Troubleshooting

### Issue: "Not enough images in cluster"
**Solution:** Reduce `--top_k_hard` or increase data

### Issue: "Memory error during generation"
**Solution:** 
```bash
# Process fewer clusters at once or reduce batch size
python src/augmentations/charm/generate_charm_dataset.py --batch_size 8
```

### Issue: "Training is slow"
**Solution:**
- Check GPU usage: `nvidia-smi`
- Increase batch size: `--batch 128` (if memory allows)
- Use faster CPU loading with `--workers 8`

---

## 📝 Paper Info

**Title:** CHARM: Cluster-Aware Hard Example Mining for Camouflaged Object Detection

**Method:** Data Augmentation

**Venue Target:** ACCV 2026 (Rank B, Top-tier)

**Key Innovation:** First to use cluster structure for hard negative mining in COD

---

## 🔗 References

- COD10K Dataset: [Paper](https://arxiv.org/abs/2104.02501)
- YOLOv8: [GitHub](https://github.com/ultralytics/ultralytics)
- Hard Example Mining: [FaceNet](https://arxiv.org/abs/1503.03832)
- CamoFA: [WACV 2025](https://arxiv.org/abs/2308.15660)

---

## 👤 Citation

```bibtex
@article{charm2026,
  title={CHARM: Cluster-Aware Hard Example Mining for Camouflaged Object Detection},
  author={Anonymous},
  journal={Submitted to ACCV},
  year={2026}
}
```

---

**Status:** ✅ Ready to train  
**Last Updated:** May 16, 2026  
**Framework:** PyTorch + YOLOv8
