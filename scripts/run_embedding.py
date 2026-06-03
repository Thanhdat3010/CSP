#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSP Step 1 — Semantic Embedding & Compression.

Downloads dataset from Google Drive (if needed), extracts DINOv2 embeddings,
and compresses them via PCA.

Usage:
    python scripts/run_embedding.py --data-dir ./data/COD10K --output-dir ./outputs/embeddings
    python scripts/run_embedding.py --download --drive-id 1NAfyqXkxYatSkfoBwGRe3CW-rI9YfaIk
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csp import config
from csp.utils import setup_logging
from csp.embedding import extract_embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSP Step 1: Semantic Embedding & PCA Compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the dataset root directory.")
    parser.add_argument("--output-dir", type=str, default="./outputs/embeddings",
                        help="Directory to save embedding outputs.")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset from Google Drive before processing.")
    parser.add_argument("--drive-id", type=str, default=config.DRIVE_FILE_ID,
                        help="Google Drive file ID for the dataset zip.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for DINOv2 inference.")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of DataLoader workers.")
    parser.add_argument("--pca-variance", type=float, default=config.PCA_VARIANCE_RATIO,
                        help="PCA variance retention ratio (0-1).")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    drive_id = args.drive_id if args.download else None

    result = extract_embeddings(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        drive_file_id=drive_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pca_variance=args.pca_variance,
    )

    print(f"\n✅ Embedding complete!")
    print(f"   Images processed: {len(result['image_paths'])}")
    print(f"   PCA dimensions:   {result['V_FINAL'].shape[1]}")
    print(f"   Saved to:         {args.output_dir}")


if __name__ == "__main__":
    main()
