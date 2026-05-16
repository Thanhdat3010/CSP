import os
import random
import shutil
from glob import glob

import cv2
import yaml
from tqdm import tqdm


def load_label_lines(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, "r", encoding="utf-8") as f:
        return [line for line in f.readlines() if line.strip()]


def write_label_lines(label_path, lines):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        if lines:
            f.writelines(lines)


def sample_lambda(alpha):
    if alpha <= 0:
        return 1.0
    return random.betavariate(alpha, alpha)


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
                write_label_lines(os.path.join(out_lbl_dir, label_name), [])


def generate_mixup_dataset(base_dir, output_dir, mixup_alpha=0.4, seed=42):
    random.seed(seed)

    base_train_img_dir = os.path.join(base_dir, "train", "images")
    base_train_lbl_dir = os.path.join(base_dir, "train", "labels")

    base_train_imgs = sorted(glob(os.path.join(base_train_img_dir, "*.jpg")))

    if not base_train_imgs:
        raise FileNotFoundError(f"No train images found in {base_train_img_dir}")

    base_count = len(base_train_imgs)

    out_train_img_dir = os.path.join(output_dir, "train", "images")
    out_train_lbl_dir = os.path.join(output_dir, "train", "labels")
    os.makedirs(out_train_img_dir, exist_ok=True)
    os.makedirs(out_train_lbl_dir, exist_ok=True)

    print(f"Base train images: {base_count}")

    # Copy all originals
    for img_path in tqdm(base_train_imgs, desc="Copying originals"):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(base_train_lbl_dir, label_name)

        shutil.copy2(img_path, os.path.join(out_train_img_dir, img_name))
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(out_train_lbl_dir, label_name))
        else:
            write_label_lines(os.path.join(out_train_lbl_dir, label_name), [])

    # Generate mixup: each original blended once with random partner
    mixup_index = 0
    for img_path1 in tqdm(base_train_imgs, desc="Generating mixup (1 per original)"):
        # Select random partner
        img_path2 = random.choice(base_train_imgs)
        
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        if img1 is None or img2 is None:
            continue

        h, w = img1.shape[:2]
        if img2.shape[:2] != (h, w):
            img2 = cv2.resize(img2, (w, h))

        # Sample lambda from Beta distribution
        lam = sample_lambda(mixup_alpha)
        mix_img = cv2.addWeighted(img1, lam, img2, 1.0 - lam, 0)

        mix_name = f"mixup_{mixup_index:06d}.jpg"
        mix_label = f"mixup_{mixup_index:06d}.txt"
        mixup_index += 1

        out_img_path = os.path.join(out_train_img_dir, mix_name)
        if not cv2.imwrite(out_img_path, mix_img):
            continue

        lbl1 = os.path.join(base_train_lbl_dir, os.path.splitext(os.path.basename(img_path1))[0] + ".txt")
        lbl2 = os.path.join(base_train_lbl_dir, os.path.splitext(os.path.basename(img_path2))[0] + ".txt")

        lines = load_label_lines(lbl1) + load_label_lines(lbl2)
        write_label_lines(os.path.join(out_train_lbl_dir, mix_label), lines)

    copy_validation_split(base_dir, output_dir)

    # Update data.yaml
    base_yaml = os.path.join(base_dir, "data.yaml")
    if os.path.exists(base_yaml):
        with open(base_yaml, "r", encoding="utf-8") as f:
            data_yaml = yaml.safe_load(f)
        data_yaml["path"] = output_dir
        data_yaml["train"] = "train/images"
        data_yaml["val"] = "val/images"

        with open(os.path.join(output_dir, "data.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, default_flow_style=False)

    total_train = base_count + mixup_index
    print(f"✅ Mixup dataset ready: {output_dir}")
    print(f"   Original images: {base_count}")
    print(f"   Mixup generated: {mixup_index}")
    print(f"   Total train images: {total_train}")


if __name__ == "__main__":
    BASE_DIR = r"d:\Code\CSP\data\COD10K-datasets"
    OUTPUT_DIR = r"d:\Code\CSP\data\COD10K-MIXUP"

    generate_mixup_dataset(BASE_DIR, OUTPUT_DIR)
