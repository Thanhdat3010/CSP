#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSP Step 6 — Unified Dataset Packager.

Merges original and synthetic datasets into a final YOLO-ready zip.

Usage:
    python scripts/run_packaging.py \\
        --synth-dir ./outputs/synthesized \\
        --original-dir ./data/COD10K \\
        --output ./outputs/COD10K-AUG-OURS
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csp import config
from csp.utils import setup_logging
from csp.packaging import package_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSP Step 6: Unified Dataset Packager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synth-dir", type=str, required=True,
                        help="Directory containing synthesized images.")
    parser.add_argument("--original-dir", type=str, required=True,
                        help="Root of the original dataset.")
    parser.add_argument("--output", type=str, default="./outputs/COD10K-AUG-OURS",
                        help="Output zip path (without .zip extension).")
    parser.add_argument("--max-workers", type=int, default=config.PACKAGE_MAX_WORKERS,
                        help="Number of threads for file copying.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    zip_path = package_dataset(
        synth_dir=args.synth_dir,
        original_dir=args.original_dir,
        output_path=args.output,
        max_workers=args.max_workers,
    )

    print(f"\n✅ Packaging complete!")
    print(f"   Output: {zip_path}")


if __name__ == "__main__":
    main()
