#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSP Step 4 — Universal Background Ingestion.

Scans all available backgrounds, computes empty-space availability,
and matches external images to semantic habitats.

Usage:
    python scripts/run_ingestion.py \\
        --partitioned-dir ./outputs/partitioned/CSP_Partitioned_Dataset \\
        --centroids ./outputs/partitioned/habitat_centroids.npy \\
        --new-data ./data/COD10K/train/non-camo \\
        --output ./outputs/environment_catalog.json
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csp import config
from csp.utils import setup_logging
from csp.ingestion import ingest_backgrounds


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSP Step 4: Universal Background Ingestion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--partitioned-dir", type=str, required=True,
                        help="Root of the CSP_Partitioned_Dataset.")
    parser.add_argument("--centroids", type=str, required=True,
                        help="Path to habitat_centroids.npy.")
    parser.add_argument("--new-data", type=str, required=True,
                        help="Path to the custom background dataset folder (containing image/ and label/).")
    parser.add_argument("--output", type=str, default="./outputs/environment_catalog.json",
                        help="Output path for the environment catalog JSON.")
    parser.add_argument("--rho-max", type=float, default=config.RHO_MAX,
                        help="Maximum background saturation threshold.")
    parser.add_argument("--sim-threshold", type=float, default=config.SIMILARITY_THRESHOLD,
                        help="Minimum cosine similarity for habitat matching.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for DINOv2 inference.")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    catalog = ingest_backgrounds(
        partitioned_dir=args.partitioned_dir,
        centroids_path=args.centroids,
        new_data=args.new_data,
        output_path=args.output,
        rho_max=args.rho_max,
        sim_threshold=args.sim_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    total_bgs = sum(len(v) for v in catalog.values())
    print(f"\n✅ Ingestion complete!")
    print(f"   Total backgrounds: {total_bgs}")
    print(f"   Total habitats:    {len(catalog)}")
    print(f"   Saved to:          {args.output}")


if __name__ == "__main__":
    main()
