#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSP Step 2 — Optimal Habitat Partitioning & Dataset Routing.

Clusters images into semantic habitats using Optuna-optimized Ward's linkage,
then routes image/label/mask triplets into cluster directories.

Usage:
    python scripts/run_partitioning.py \\
        --embeddings ./outputs/embeddings \\
        --data-dir ./data/COD10K \\
        --output-dir ./outputs/partitioned
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csp import config
from csp.utils import setup_logging
from csp.partitioning import run_partitioning


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSP Step 2: Habitat Partitioning & Dataset Routing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Directory containing saved embeddings from Step 1.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the original dataset root.")
    parser.add_argument("--output-dir", type=str, default="./outputs/partitioned",
                        help="Output directory for partitioned dataset.")
    parser.add_argument("--n-trials", type=int, default=config.OPTUNA_N_TRIALS,
                        help="Number of Optuna optimization trials.")
    parser.add_argument("--k-min", type=int, default=config.OPTUNA_K_MIN,
                        help="Minimum number of clusters to search.")
    parser.add_argument("--k-max", type=int, default=config.OPTUNA_K_MAX,
                        help="Maximum number of clusters to search.")
    parser.add_argument("--max-workers", type=int, default=config.MAX_WORKERS,
                        help="Number of threads for file routing.")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    result = run_partitioning(
        embeddings_dir=args.embeddings,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        k_min=args.k_min,
        k_max=args.k_max,
        max_workers=args.max_workers,
        seed=args.seed,
    )

    print(f"\n✅ Partitioning complete!")
    print(f"   Optimal k:        {result['optimal_k']}")
    print(f"   Best DB Score:    {result['best_score']:.4f}")
    print(f"   Missing labels:   {result['missing_labels']}")
    print(f"   Missing masks:    {result['missing_masks']}")
    print(f"   Output:           {result['partitioned_dir']}")


if __name__ == "__main__":
    main()
