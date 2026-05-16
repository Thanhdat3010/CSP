"""
CSP-Diff: Cluster-Guided Diffusion Background Regeneration for COD

Pipeline:
  1. Read CSP clusters from COD10K_Partitioning (cluster_* subdirs)
  2. Within each cluster, compute descriptors → find hard negatives
  3. For each image, use hard negative as IP-Adapter reference for
     SD Inpainting to regenerate background while preserving object (BBox)
  4. Output: originals + augmented + val + data.yaml

Usage (local or Colab with A100):
  python src/augmentations/cspdiff/generate_cspdiff_dataset.py \
      --data_dir data/COD10K_Partitioning \
      --output data/COD10K-CSPDiff \
      --num_variants 2
"""

import argparse, os, shutil
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import cv2, yaml, torch


# ── YOLO label parsing ─────────────────────────────────────────────────

def parse_yolo(lbl_path, w, h):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    for line in open(lbl_path):
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = map(float, p[1:5])
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        boxes.append((c, x1, y1, x2, y2))
    return boxes


# ── Inpainting mask ────────────────────────────────────────────────────

def make_mask(w, h, boxes, margin=20, blur=51):
    """White=inpaint (background), black=keep (object BBox+margin)."""
    m = np.ones((h, w), np.float32)
    for _, x1, y1, x2, y2 in boxes:
        m[max(0, y1 - margin):min(h, y2 + margin),
          max(0, x1 - margin):min(w, x2 + margin)] = 0.0
    k = blur if blur % 2 == 1 else blur + 1
    m = cv2.GaussianBlur(m, (k, k), 0)
    return Image.fromarray((m * 255).clip(0, 255).astype(np.uint8), "L")


# ── Image descriptor (for hard-negative mining) ───────────────────────

def descriptor(path):
    """Color histogram (24-d) + gradient histogram (8-d) = 32-d vector."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros(32)
    feats = []
    for c in range(3):
        h = cv2.calcHist([img], [c], None, [8], [0, 256]).flatten()
        feats.append(h / (h.sum() + 1e-8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    gh = np.histogram(mag, bins=8, range=(0, mag.max() + 1e-8))[0].astype(float)
    feats.append(gh / (gh.sum() + 1e-8))
    return np.concatenate(feats)


# ── Hard negative selection within a cluster ───────────────────────────

def find_hard_negs(descs, k=2):
    """For each image in the cluster, return indices of k nearest neighbors."""
    n = len(descs)
    mapping = {}
    for i in range(n):
        dists = [(j, np.linalg.norm(descs[i] - descs[j]))
                 for j in range(n) if j != i]
        dists.sort(key=lambda x: x[1])
        mapping[i] = [j for j, _ in dists[:k]]
    return mapping


# ── SD helpers ─────────────────────────────────────────────────────────

def sd_resize(w, h, mx=512):
    s = min(mx / w, mx / h, 1.0)
    return max(8, int(w * s) // 8 * 8), max(8, int(h * s) // 8 * 8)


# ── Main ───────────────────────────────────────────────────────────────

def generate(args):
    data_dir = args.data_dir
    out_dir = args.output
    K = args.num_variants

    # Output dirs
    d = {}
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            p = os.path.join(out_dir, sp, sub)
            os.makedirs(p, exist_ok=True)
            d[f"{sp}_{sub}"] = p

    # ── Load pipeline ──────────────────────────────────────────────────
    print("⏳ Loading SD Inpainting + IP-Adapter …")
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.bin",
    )
    pipe.set_ip_adapter_scale(args.ip_scale)
    pipe.set_progress_bar_config(disable=True)
    print("✅ Pipeline ready.\n")

    # ── Discover CSP clusters ──────────────────────────────────────────
    train_img_root = os.path.join(data_dir, "train", "images")
    train_lbl_root = os.path.join(data_dir, "train", "labels")

    clusters = sorted([
        c for c in os.listdir(train_img_root)
        if os.path.isdir(os.path.join(train_img_root, c))
    ])
    print(f"Found {len(clusters)} CSP clusters\n")

    total_aug = 0

    for cluster in clusters:
        img_cdir = os.path.join(train_img_root, cluster)
        lbl_cdir = os.path.join(train_lbl_root, cluster)

        files = sorted([
            f for f in os.listdir(img_cdir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if len(files) < 2:
            # Copy originals only, not enough peers for hard-neg
            for f in files:
                shutil.copy2(os.path.join(img_cdir, f),
                             os.path.join(d["train_images"], f))
                lf = Path(f).stem + ".txt"
                lp = os.path.join(lbl_cdir, lf)
                if os.path.exists(lp):
                    shutil.copy2(lp, os.path.join(d["train_labels"], lf))
            continue

        # Compute descriptors
        paths = [os.path.join(img_cdir, f) for f in files]
        descs = np.array([descriptor(p) for p in paths])

        # Hard negatives
        hn = find_hard_negs(descs, k=K)

        for i, fname in enumerate(tqdm(files, desc=cluster, leave=False)):
            stem = Path(fname).stem
            lbl_name = stem + ".txt"
            img_path = os.path.join(img_cdir, fname)
            lbl_path = os.path.join(lbl_cdir, lbl_name)

            # Always copy original
            shutil.copy2(img_path, os.path.join(d["train_images"], fname))
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path,
                             os.path.join(d["train_labels"], lbl_name))

            # Skip augmentation if no label
            if not os.path.exists(lbl_path):
                continue

            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            boxes = parse_yolo(lbl_path, iw, ih)
            if not boxes:
                continue

            mask = make_mask(iw, ih, boxes, margin=args.margin)
            sw, sh = sd_resize(iw, ih)
            img_sd = img.resize((sw, sh), Image.LANCZOS)
            mask_sd = mask.resize((sw, sh), Image.LANCZOS)

            for v, neg_i in enumerate(hn.get(i, [])):
                ref = Image.open(paths[neg_i]).convert("RGB")
                ref_sd = ref.resize((sw, sh), Image.LANCZOS)

                seed = (args.seed + hash(fname) + v) % (2 ** 32)
                gen = torch.Generator("cuda").manual_seed(seed)

                out_img = pipe(
                    prompt=args.prompt,
                    image=img_sd,
                    mask_image=mask_sd,
                    ip_adapter_image=ref_sd,
                    strength=args.strength,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    generator=gen,
                ).images[0]

                # Composite: guarantee object preservation
                out_full = out_img.resize((iw, ih), Image.LANCZOS)
                mn = np.array(mask, np.float32) / 255.0
                m3 = np.stack([mn] * 3, axis=-1)
                comp = (np.array(img) * (1 - m3)
                        + np.array(out_full) * m3)
                comp = comp.clip(0, 255).astype(np.uint8)

                aug_f = f"{stem}_cspdiff_v{v}.jpg"
                aug_l = f"{stem}_cspdiff_v{v}.txt"
                Image.fromarray(comp).save(
                    os.path.join(d["train_images"], aug_f), quality=95)
                shutil.copy2(lbl_path,
                             os.path.join(d["train_labels"], aug_l))
                total_aug += 1

    print(f"\n✅ Generated {total_aug} augmented training images")

    # ── Copy val ───────────────────────────────────────────────────────
    print("── Copying validation set ──")
    for sub in ("images", "labels"):
        src = os.path.join(data_dir, "val", sub)
        if not os.path.isdir(src):
            continue
        # val may also have cluster subdirs or be flat
        for root, dirs_list, fnames in os.walk(src):
            for f in fnames:
                shutil.copy2(os.path.join(root, f),
                             os.path.join(d[f"val_{sub}"], f))

    # ── data.yaml ──────────────────────────────────────────────────────
    src_yaml = os.path.join(data_dir, "data.yaml")
    if os.path.exists(src_yaml):
        with open(src_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["path"] = os.path.abspath(out_dir)
        with open(os.path.join(out_dir, "data.yaml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    # ── Summary ────────────────────────────────────────────────────────
    nt = len(os.listdir(d["train_images"]))
    nv = len(os.listdir(d["val_images"]))
    print(f"\n{'=' * 50}")
    print(f"✅ CSP-Diff dataset: {out_dir}")
    print(f"   Train : {nt} (originals + augmented)")
    print(f"   Val   : {nv}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="CSP-Diff: Cluster-Guided Diffusion Background Augmentation")
    pa.add_argument("--data_dir", default="data/COD10K_Partitioning")
    pa.add_argument("--output", default="data/COD10K-CSPDiff")
    pa.add_argument("--num_variants", type=int, default=2)
    pa.add_argument("--margin", type=int, default=20)
    pa.add_argument("--strength", type=float, default=0.75)
    pa.add_argument("--ip_scale", type=float, default=0.6)
    pa.add_argument("--prompt", default="natural environment, photorealistic")
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()
    generate(args)
