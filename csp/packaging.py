# -*- coding: utf-8 -*-
"""
CSP Phase 2 — Unified Dataset Packager.

Corresponds to Cell 9 of the original notebook:
1. Merge original (camo + non-camo) training data
2. Merge synthesized data
3. Copy val + data.yaml
4. Zip to final output
"""

import os
import shutil
import logging
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .utils import ensure_dir

logger = logging.getLogger("csp.packaging")


def package_dataset(
    synth_dir: str,
    original_dir: str,
    output_path: str,
    max_workers: int = config.PACKAGE_MAX_WORKERS,
):
    """Merge original + synthetic datasets into a YOLO-ready package.

    Args:
        synth_dir: Directory containing synthesized images/labels/masks.
        original_dir: Root of the original dataset.
        output_path: Output zip path (without .zip extension — added automatically).
        max_workers: Number of threads for parallel file copying.

    Returns:
        Path to the created zip file.
    """
    staging_dir = os.path.join(os.path.dirname(output_path), "YOLO_Export")

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    ensure_dir(staging_dir)

    # Target structure
    train_img_dir = os.path.join(staging_dir, "train", "image")
    train_lbl_dir = os.path.join(staging_dir, "train", "label")
    train_msk_dir = os.path.join(staging_dir, "train", "mask")
    ensure_dir(train_img_dir)
    ensure_dir(train_lbl_dir)
    ensure_dir(train_msk_dir)

    # Copy val + data.yaml
    logger.info("Copying 'val' directory and 'data.yaml'...")
    val_src = os.path.join(original_dir, "val")
    if os.path.exists(val_src):
        shutil.copytree(val_src, os.path.join(staging_dir, "val"))
    else:
        logger.warning("'val' directory not found in source.")

    yaml_src = os.path.join(original_dir, "data.yaml")
    if os.path.exists(yaml_src):
        shutil.copy(yaml_src, os.path.join(staging_dir, "data.yaml"))

    # Build copy task list
    tasks = []

    # Queue original train data
    logger.info("Queuing original training files...")
    for subset in ["camo", "non-camo"]:
        for folder_type in ["image", "label", "mask"]:
            src_folder = os.path.join(original_dir, "train", subset, folder_type)
            dest_folder = os.path.join(staging_dir, "train", folder_type)

            if os.path.exists(src_folder):
                for filename in os.listdir(src_folder):
                    src_file = os.path.join(src_folder, filename)
                    dest_file = os.path.join(dest_folder, filename)
                    tasks.append((src_file, dest_file))

    # Queue synthesized data
    logger.info("Queuing synthesized data...")
    if os.path.exists(synth_dir):
        for filename in os.listdir(synth_dir):
            src_file = os.path.join(synth_dir, filename)

            if filename.lower().endswith((".jpg", ".jpeg")):
                dest_file = os.path.join(train_img_dir, filename)
            elif filename.lower().endswith(".txt"):
                dest_file = os.path.join(train_lbl_dir, filename)
            elif filename.lower().endswith(".png"):
                dest_file = os.path.join(train_msk_dir, filename)
            else:
                continue

            tasks.append((src_file, dest_file))

    # Execute parallel copy
    def copy_task(args):
        src, dst = args
        if os.path.exists(src):
            shutil.copy(src, dst)

    logger.info("Merging %d files...", len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_task, t) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Merging & Routing Files"):
            pass

    # Zip
    logger.info("Creating final zip archive...")
    zip_path = shutil.make_archive(output_path, "zip", staging_dir)

    # Cleanup staging
    shutil.rmtree(staging_dir)

    logger.info("Export complete: %s", zip_path)
    return zip_path
