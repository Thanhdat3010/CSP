import os
import random
import shutil
from glob import glob

import cv2
import yaml
from tqdm import tqdm


def load_labels(label_path):
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            labels.append((int(float(cls)), float(x), float(y), float(w), float(h)))
    return labels


def write_labels(label_path, labels):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def sample_lambda(alpha):
    if alpha <= 0:
        return 1.0
    return random.betavariate(alpha, alpha)


def yolo_to_xyxy(label, img_w, img_h):
    cls, x, y, w, h = label
    x1 = (x - w / 2.0) * img_w
    y1 = (y - h / 2.0) * img_h
    x2 = (x + w / 2.0) * img_w
    y2 = (y + h / 2.0) * img_h
    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, img_w, img_h):
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    x = (x1 + x2) / 2.0 / img_w
    y = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cls, x, y, w, h


def copy_validation_split(base_dir, output_dir):
    for split in ["val"]:
        out_img_dir = os.path.join(output_dir, split, "images")
        out_lbl_dir = os.path.join(output_dir, split, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        img_paths = glob(os.path.join(base_dir, split, "images", "*.jpg"))
        for img_path in tqdm(img_paths, desc=f"Copying {split}"):
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(base_dir, split, "labels", label_name)

            shutil.copy2(img_path, os.path.join(out_img_dir, img_name))
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(out_lbl_dir, label_name))
            else:
                write_labels(os.path.join(out_lbl_dir, label_name), [])


def cutmix_labels(labels1, labels2, cut_rect, img_w, img_h):
    cx1, cy1, cx2, cy2 = cut_rect
    mixed = []

    for label in labels1:
        cls, x, y, w, h = label
        px = x * img_w
        py = y * img_h
        if not (cx1 <= px <= cx2 and cy1 <= py <= cy2):
            mixed.append(label)

    for label in labels2:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(label, img_w, img_h)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if cx1 <= cx <= cx2 and cy1 <= cy <= cy2:
            nx1 = max(x1, cx1)
            ny1 = max(y1, cy1)
            nx2 = min(x2, cx2)
            ny2 = min(y2, cy2)
            new_label = xyxy_to_yolo(cls, nx1, ny1, nx2, ny2, img_w, img_h)
            if new_label is not None:
                mixed.append(new_label)

    return mixed


def generate_cutmix_dataset(base_dir, smm_dir, output_dir, mixup_alpha=0.4, seed=42):
    random.seed(seed)

    base_train_img_dir = os.path.join(base_dir, "train", "images")
    base_train_lbl_dir = os.path.join(base_dir, "train", "labels")

    smm_train_img_dir = os.path.join(smm_dir, "train", "images")

    base_train_imgs = sorted(glob(os.path.join(base_train_img_dir, "*.jpg")))
    smm_train_imgs = sorted(glob(os.path.join(smm_train_img_dir, "*.jpg")))

    if not base_train_imgs:
        raise FileNotFoundError(f"No train images found in {base_train_img_dir}")
    if not smm_train_imgs:
        raise FileNotFoundError(f"No train images found in {smm_train_img_dir}")

    base_count = len(base_train_imgs)
    target_count = len(smm_train_imgs)

    print(f"Base train images: {base_count}")
    print(f"SMM train images (target): {target_count}")

    out_train_img_dir = os.path.join(output_dir, "train", "images")
    out_train_lbl_dir = os.path.join(output_dir, "train", "labels")
    os.makedirs(out_train_img_dir, exist_ok=True)
    os.makedirs(out_train_lbl_dir, exist_ok=True)

    if base_count > target_count:
        raise ValueError(
            "Base train images exceed SMM target count; cannot keep totals equal "
            "while preserving all originals."
        )

    current_count = 0

    for img_path in tqdm(base_train_imgs, desc="Copying originals"):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(base_train_lbl_dir, label_name)

        shutil.copy2(img_path, os.path.join(out_train_img_dir, img_name))
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(out_train_lbl_dir, label_name))
        else:
            write_labels(os.path.join(out_train_lbl_dir, label_name), [])

        current_count += 1

    cutmix_index = 0
    cutmix_needed = max(0, target_count - current_count)
    cutmix_bar = tqdm(total=cutmix_needed, desc="Generating cutmix")
    while current_count < target_count:
        img_path1, img_path2 = random.sample(base_train_imgs, 2)
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        if img1 is None or img2 is None:
            continue

        h, w = img1.shape[:2]
        if img2.shape[:2] != (h, w):
            img2 = cv2.resize(img2, (w, h))

        lam = sample_lambda(mixup_alpha)
        cut_w = int(w * (1.0 - lam) ** 0.5)
        cut_h = int(h * (1.0 - lam) ** 0.5)
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, x1 + cut_w)
        y2 = min(h, y1 + cut_h)

        img1[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

        lbl1 = os.path.join(base_train_lbl_dir, os.path.splitext(os.path.basename(img_path1))[0] + ".txt")
        lbl2 = os.path.join(base_train_lbl_dir, os.path.splitext(os.path.basename(img_path2))[0] + ".txt")
        labels1 = load_labels(lbl1)
        labels2 = load_labels(lbl2)

        mixed_labels = cutmix_labels(labels1, labels2, (x1, y1, x2, y2), w, h)

        out_name = f"cutmix_{cutmix_index:06d}.jpg"
        out_label = f"cutmix_{cutmix_index:06d}.txt"
        cutmix_index += 1

        out_img_path = os.path.join(out_train_img_dir, out_name)
        if not cv2.imwrite(out_img_path, img1):
            continue

        write_labels(os.path.join(out_train_lbl_dir, out_label), mixed_labels)

        current_count += 1
        cutmix_bar.update(1)

    cutmix_bar.close()

    copy_validation_split(base_dir, output_dir)

    base_yaml = os.path.join(base_dir, "data.yaml")
    if os.path.exists(base_yaml):
        with open(base_yaml, "r", encoding="utf-8") as f:
            data_yaml = yaml.safe_load(f)
        data_yaml["path"] = output_dir
        data_yaml["train"] = "train/images"
        data_yaml["val"] = "val/images"

        with open(os.path.join(output_dir, "data.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, default_flow_style=False)

    print(f"CutMix dataset ready: {output_dir}")
    print(f"Train images (target): {target_count}")


if __name__ == "__main__":
    BASE_DIR = r"d:\Code\CSP\data\COD10K-datasets"
    SMM_DIR = r"d:\Code\CSP\data\COD10K-SMM"
    OUTPUT_DIR = r"d:\Code\CSP\data\COD10K-CUTMIX"

    generate_cutmix_dataset(BASE_DIR, SMM_DIR, OUTPUT_DIR)
