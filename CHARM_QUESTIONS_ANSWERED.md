# CHARM Augmentation - Q&A

## Q1: Tại sao augment ra nhiều ảnh train? (3000 → ?)

### Đáp án Chi Tiết

**Tập gốc:** ~3,000 ảnh (chỉ camouflaged objects, không non-camouflaged)

**Quá trình augmentation:**

```
Mỗi ảnh gốc → 4 ảnh trong training set:
  1. Original (giữ nguyên)
  2. CHARM Variant 1 (object gốc + background từ hard negative 1)
  3. CHARM Variant 2 (object gốc + background từ hard negative 2)
  4. CHARM Variant 3 (object gốc + background từ hard negative 3)

Công thức:
  Total = Original_images + (Original_images × top_k_hard)
  Total = 3,000 + (3,000 × 3)
  Total = 3,000 + 9,000
  Total = 12,000 training images
```

**Tại sao nhiều thế:**

- ✅ **Preservation:** Mỗi ảnh gốc được giữ lại y nguyên (confidence không bị mất)
- ✅ **Diversity:** 3 variants khác nhau → 3 bối cảnh khác nhau
- ✅ **Hard Negatives:** Mỗi variant dùng background từ ảnh khó (chứ không random)
- ✅ **Within-Cluster:** Tất cả hard negatives từ cùng cluster (cùng loại camouflage)

**Nếu dùng tham số khác:**

```
top_k_hard=1  → 3,000 + 3,000 = 6,000 images (2x)
top_k_hard=2  → 3,000 + 6,000 = 9,000 images (3x)
top_k_hard=3  → 3,000 + 9,000 = 12,000 images (4x) ← Default
top_k_hard=5  → 3,000 + 15,000 = 18,000 images (6x)
```

**So sánh với phương pháp khác:**

| Phương pháp | 3000 gốc → | Nhận xét |
|------------|-----------|---------|
| Không augment | 3,000 | Quá ít data, dễ overfit |
| Mixup/CutMix | 6,000 | 2x, nhưng giáp hợp lí |
| Generic morphing | 6,000 | 2x, nhưng mất object boundary |
| **CHARM** | **12,000** | **4x, giữ object + hard background** |

---

## Q2: Xóa script SMM, tạo script visualize CHARM

### ✅ Đã hoàn thành

**Xóa:**
- ❌ `scripts/verify_smm.py` (không tồn tại nên k cần xóa)
- ❌ Tham chiếu SMM trong `check_dataset.py` → đã sửa

**Tạo mới:**
- ✅ `scripts/visualize_charm.py` - Hiển thị CHARM method
- ✅ `scripts/check_dataset.py` - Update để support CHARM
- ✅ `scripts/count_data.py` - Count ảnh per cluster

### Các script mới

#### 1. **visualize_charm.py** - Hiển thị phương pháp CHARM
```bash
python scripts/visualize_charm.py \
    --dataset data/COD10K_Partitioning/train \
    --output charm_visualization.jpg \
    --samples 3
```

**Output:** Hình collage hiển thị:
- Column 1: Original image
- Column 2: Variant 1 (hard negative background 1)
- Column 3: Variant 2 (hard negative background 2)
- Column 4: Variant 3 (hard negative background 3)

#### 2. **check_dataset.py** - Verify dataset integrity
```bash
python scripts/check_dataset.py --dataset data/COD10K-CHARM
```

**Output:**
```
📊 CHARM Dataset Integrity Check
📁 TRAIN SET
   Images: 12,000
   Labels: 12,000
   Match: ✅

📊 AUGMENTATION ANALYSIS
   Original images: 3,000
   Augmented images: 9,000
   Variants per original: 3.0x
```

#### 3. **count_data.py** - Tính toán augmentation
```bash
python scripts/count_data.py --partition data/COD10K_Partitioning
```

**Output:**
```
💡 NOTE: With top_k_hard=3 (default):
   3,000 originals → 12,000 total train images
   (1 original + 3 augmented variants per image)
```

---

## Q3: Lệnh chạy để augment ra data hoàn chỉnh

### ✅ Cách nhanh nhất: 1 lệnh duy nhất

```bash
cd d:\Code\CSP
.\augment_charm_now.bat
```

**Lệnh này sẽ tự động:**
1. ✅ Count dataset (bao nhiêu ảnh per cluster)
2. ✅ Generate CHARM dataset (12,000 training images)
3. ✅ Check integrity (verify all images matched with labels)
4. ✅ Visualize method (tạo charm_visualization.jpg)

**Output cuối:**
```
✅ CHARM AUGMENTATION COMPLETE!

📊 Generated Dataset:
   Location: data/COD10K-CHARM/
   train/images/  - 12,000 augmented images
   train/labels/  - 12,000 labels
   val/images/    - ~500 validation images
   val/labels/    - ~500 validation labels
   data.yaml      - YOLO configuration
```

---

### Cách manual (nếu cần tuning):

**Step 1: Count data**
```bash
python scripts/count_data.py --partition data/COD10K_Partitioning
```

**Step 2: Generate với tuning**
```bash
python src/augmentations/charm/generate_charm_dataset.py ^
    --partition_dir data/COD10K_Partitioning ^
    --output data/COD10K-CHARM ^
    --margin 10 ^
    --smooth_kernel 15 ^
    --top_k_hard 3 ^
    --split_ratio 0.8
```

**Parameters có thể tuning:**
- `--top_k_hard`: Number of hard negatives per image (1-5, default 3)
- `--margin`: Safety buffer around object in pixels (5-20, default 10)
- `--smooth_kernel`: Gaussian blur kernel (5-21, odd numbers only, default 15)
- `--split_ratio`: Train/val split (0.7-0.9, default 0.8)

**Step 3: Verify**
```bash
python scripts/check_dataset.py --dataset data/COD10K-CHARM
```

**Step 4: Visualize**
```bash
python scripts/visualize_charm.py ^
    --dataset data/COD10K_Partitioning/train ^
    --output charm_visualization.jpg ^
    --samples 3
```

---

## Tóm tắt

| Câu hỏi | Đáp án |
|--------|--------|
| **Tại sao 3000 → 12000?** | Mỗi ảnh → 4 variant (1 orig + 3 hard neg). 3000 × 4 = 12,000 |
| **SMM script?** | ✅ Xóa rồi, tạo 3 script mới cho CHARM |
| **Lệnh chạy?** | `.\augment_charm_now.bat` (all-in-one) hoặc manual steps |

---

## 🚀 Chạy ngay

```bash
cd d:\Code\CSP
.\augment_charm_now.bat
```

**Sẽ hoàn thành trong ~20 phút (tuỳ theo tốc độ disk).**

✅ Ready for Colab training!
