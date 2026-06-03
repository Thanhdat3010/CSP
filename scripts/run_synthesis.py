#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSP Step 5 — Master Synthesis Engine.

Generates synthetic camouflaged images by pasting objects from the Dictionary
into backgrounds from the Catalog, with full physics validation and harmonization.

Usage:
    python scripts/run_synthesis.py \\
        --catalog ./outputs/environment_catalog.json \\
        --dictionary ./outputs/latent_affordance_dictionary.json \\
        --data-dir ./data/COD10K \\
        --output-dir ./outputs/synthesized
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csp import config
from csp.utils import setup_logging
from csp.synthesis import synthesize


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSP Step 5: Master Synthesis Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--catalog", type=str, required=True,
                        help="Path to environment_catalog.json.")
    parser.add_argument("--dictionary", type=str, required=True,
                        help="Path to latent_affordance_dictionary.json.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root of original dataset (for masks/labels).")
    parser.add_argument("--output-dir", type=str, default="./outputs/synthesized",
                        help="Output directory for synthesized images.")
    parser.add_argument("--batch-size", type=int, default=config.SYNTHESIS_BATCH_SIZE,
                        help="Batch size for MiDaS inference.")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of DataLoader workers.")
    parser.add_argument("--max-retries", type=int, default=config.MAX_RETRIES,
                        help="Max placement attempts per object.")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    count = synthesize(
        catalog_path=args.catalog,
        dictionary_path=args.dictionary,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_retries=args.max_retries,
        seed=args.seed,
    )

    print(f"\n✅ Synthesis complete!")
    print(f"   Images generated: {count}")
    print(f"   Saved to:         {args.output_dir}")


if __name__ == "__main__":
    main()
