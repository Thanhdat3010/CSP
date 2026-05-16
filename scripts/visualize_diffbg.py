"""Quick visual check for CSP-Diff augmented images."""
import os, sys, random
from PIL import Image
from pathlib import Path


def visualize(data_dir, n=4, out="cspdiff_preview.jpg"):
    img_dir = os.path.join(data_dir, "train", "images")
    if not os.path.isdir(img_dir):
        print(f"Not found: {img_dir}"); return

    all_f = sorted(os.listdir(img_dir))
    originals = [f for f in all_f if "_cspdiff_" not in f]
    random.seed(42)
    picks = random.sample(originals, min(n, len(originals)))

    rows = []
    for orig in picks:
        stem = Path(orig).stem
        variants = [f for f in all_f if f.startswith(stem + "_cspdiff_")][:3]
        imgs = [Image.open(os.path.join(img_dir, orig)).convert("RGB")]
        imgs += [Image.open(os.path.join(img_dir, v)).convert("RGB") for v in variants]

        h = 256
        resized = [im.resize((int(im.width * h / im.height), h), Image.LANCZOS) for im in imgs]
        row = Image.new("RGB", (sum(r.width for r in resized), h))
        x = 0
        for r in resized:
            row.paste(r, (x, 0)); x += r.width
        rows.append(row)

    max_w = max(r.width for r in rows)
    canvas = Image.new("RGB", (max_w, sum(r.height for r in rows)), (30, 30, 30))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y)); y += r.height
    canvas.save(out, quality=95)
    print(f"✅ Preview: {out}")


if __name__ == "__main__":
    visualize(sys.argv[1] if len(sys.argv) > 1 else "data/COD10K-CSPDiff")
