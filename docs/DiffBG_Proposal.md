# CSP-Diff: Cluster-Guided Diffusion Background Regeneration for COD

## Tóm tắt 1 câu

> Dùng CSP phân hoạch ngữ nghĩa → tìm Hard Negative cùng cluster → dùng ảnh Hard Negative đó làm **reference conditioning** cho SD Inpainting để sinh lại background → Object giữ nguyên (BBox), background mới giống môi trường khó nhất trong cùng cluster, không có viền cắt.

---

## Pipeline

```
Ảnh A (dễ, cluster_5)
    │
    ├─ 1. CSP Partitioning đã gom vào cluster_5
    │
    ├─ 2. Tìm Hard Negative B trong cluster_5
    │     (ảnh có descriptor gần nhất nhưng model detect kém nhất)
    │
    ├─ 3. Tạo inpaint mask từ BBox
    │     BBox area = KEEP (object)
    │     Phần còn lại = INPAINT (background)
    │
    ├─ 4. SD Inpainting + IP-Adapter
    │     reference_image = ảnh B (Hard Negative)
    │     → Diffusion sinh background mới giống môi trường của ảnh B
    │     → Chuyển tiếp mượt mà tại biên BBox (do diffusion tự xử lý)
    │
    └─ 5. Output: Object A nguyên vẹn + Background kiểu B
           Label BBox giữ nguyên 100%
```

---

## 3 Contributions rõ ràng

### C1: Context-Preserving Constraint (Từ CSP)
Augmentation chỉ diễn ra **trong cùng cluster ngữ nghĩa**. Background mới được sinh từ reference image cùng môi trường (rừng → rừng, biển → biển). Không ai khác có cơ chế đảm bảo điều này — CamDiff dùng CLIP prompt chung chung, CamoFA trộn frequency ngẫu nhiên.

### C2: Hard Negative-Guided Generation (Mới hoàn toàn)
Thay vì sinh background ngẫu nhiên hoặc từ text prompt, ta dùng **chính một ảnh thật khó nhất** trong cùng cluster làm visual reference. Diffusion model được conditioning bởi ảnh thật → output realistic hơn rất nhiều so với text-only generation. Đây là sự kết hợp chưa ai làm: **hard negative mining + reference-guided inpainting**.

### C3: BBox-Only Seamless Augmentation (Practical novelty)
Chứng minh rằng chỉ cần BBox (không cần segmentation mask) vẫn có thể tạo ra augmented data chất lượng cao cho COD. Diffusion inpainting tự xử lý đường chuyển tiếp → loại bỏ hoàn toàn vấn đề "halo effect" mà các phương pháp copy-paste gặp phải.

---

## Tại sao đủ novelty cho ACCV rank B?

### So sánh trực tiếp với các bài đã publish:

| | CamoFA (WACV 2025) | CamDiff (2026) | **CSP-Diff (Ours)** |
|---|---|---|---|
| Kỹ thuật | Fourier swap + GAN | Latent Diffusion + CLIP | **SD Inpainting + IP-Adapter** |
| Reference image | ❌ Không | ❌ Không (text-only) | **✅ Hard Negative thật** |
| Context-aware | ❌ Random/global | ❌ CLIP prompt | **✅ CSP cluster** |
| Cần Seg Mask | ✅ Cần | Tùy variant | **❌ Chỉ cần BBox** |
| Object preservation | ⚠️ Object bị biến đổi | ⚠️ Sinh mới hoàn toàn | **✅ 100% giữ nguyên** |
| Spatial artifacts | ⚠️ Frequency artifacts | ✅ Không | **✅ Không** |

### Điểm bán hàng (selling points) cho reviewer:

1. **"Ngược chiều" với CamDiff**: CamDiff sinh ảnh mới hoàn toàn → object là giả. Ta giữ object thật, chỉ đổi background → object luôn authentic.
2. **Hard Negative là ảnh thật, không phải synthetic**: Reference image là ảnh COD10K thật sự khó detect, không phải output của GAN/Diffusion. Tính chân thực cao hơn.
3. **CSP cluster đảm bảo context**: Không có bài nào khác có cơ chế semantic partitioning chặt chẽ để ràng buộc augmentation.

---

## Ablation Study dự kiến (rất quan trọng cho paper)

| Experiment | Mục đích |
|-----------|---------|
| Baseline (no aug) | Điểm gốc |
| Random Inpainting (no cluster, no reference) | Chứng minh cluster + reference quan trọng |
| Cluster-only (random reference trong cluster) | Chứng minh hard negative selection quan trọng |
| Reference-only (hard neg nhưng cross-cluster) | Chứng minh CSP cluster quan trọng |
| **CSP-Diff (full pipeline)** | **Kết quả tốt nhất** |

Nếu mỗi dòng ablation cho kết quả tăng dần → paper cực kỳ thuyết phục.

---

## Implementation cần gì

| Component | Tool | Có sẵn? |
|-----------|------|---------|
| CSP Partitioning | COD10K_Partitioning folder | ✅ Đã có |
| Hard Negative Mining | HOG + Histogram descriptor | ✅ Code từ CHARM, tái sử dụng |
| SD Inpainting | `diffusers` library | ✅ Có sẵn |
| IP-Adapter (reference conditioning) | `ip-adapter` package | ✅ Open source |
| BBox mask creation | OpenCV | ✅ Đã code |
| YOLOv8 training | Ultralytics | ✅ Có sẵn |

**Thời gian ước tính:** 2-3 ngày code + 1 ngày gen data + 1 ngày train trên A100.

---

## Verdict

CSP-Diff kết hợp 3 thành phần (CSP + Hard Negative Mining + Reference-Guided Diffusion), mỗi thành phần đơn lẻ không mới, nhưng **sự kết hợp cụ thể này cho bài toán COD với BBox-only chưa ai làm**. Ablation study sẽ chứng minh mỗi thành phần đều cần thiết. Đây là mức novelty phù hợp cho ACCV rank B (applied method paper).
