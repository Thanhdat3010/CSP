#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSP Step 3 — Latent Affordance Dictionary.

Extracts physical properties (topography, lighting, solidity) for every
camouflaged object and builds the affordance dictionary with anomaly detection.

Usage:
    python scripts/run_dictionary.py \\
        --partitioned-dir ./outputs/partitioned/CSP_Partitioned_Dataset \\
        --output ./outputs/latent_affordance_dictionary.json
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csp import config
from csp.utils import setup_logging
from csp.dictionary import build_dictionary


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSP Step 3: Latent Affordance Dictionary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--partitioned-dir", type=str, required=True,
                        help="Root of the CSP_Partitioned_Dataset.")
    parser.add_argument("--output", type=str, default="./outputs/latent_affordance_dictionary.json",
                        help="Output path for the dictionary JSON.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for model inference.")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of DataLoader workers.")
    parser.add_argument("--contamination", type=float, default=config.ISOLATION_CONTAMINATION,
                        help="Isolation Forest contamination rate.")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    dictionary = build_dictionary(
        partitioned_dir=args.partitioned_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        contamination=args.contamination,
        seed=args.seed,
    )

    total_objects = sum(len(v["objects"]) for v in dictionary.values())
    print(f"\n✅ Dictionary built!")
    print(f"   Total objects:    {total_objects}")
    print(f"   Total clusters:   {len(dictionary)}")
    print(f"   Saved to:         {args.output}")


if __name__ == "__main__":
    main()
