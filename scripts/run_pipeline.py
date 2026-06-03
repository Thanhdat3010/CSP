#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSP — Full End-to-End Pipeline Runner.

Executes all 6 steps of the Context-Synchronized Pipeline sequentially:
  1. Semantic Embedding & PCA Compression
  2. Habitat Partitioning & Dataset Routing
  3. Latent Affordance Dictionary
  4. Background Ingestion
  5. Synthesis Engine
  6. Dataset Packaging

Usage:
    python scripts/run_pipeline.py \\
        --data-dir ./data/COD10K \\
        --output-dir ./outputs \\
        --seed 42

    # With auto-download from Google Drive:
    python scripts/run_pipeline.py \\
        --data-dir ./data/COD10K \\
        --output-dir ./outputs \\
        --download --drive-id 1NAfyqXkxYatSkfoBwGRe3CW-rI9YfaIk
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csp import config
from csp.utils import setup_logging, ensure_dir
from csp.embedding import extract_embeddings
from csp.partitioning import run_partitioning
from csp.dictionary import build_dictionary
from csp.ingestion import ingest_backgrounds
from csp.synthesis import synthesize
from csp.packaging import package_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="CSP: Full End-to-End Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core paths
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the dataset root directory.")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Root output directory for all pipeline artifacts.")

    # Download options
    parser.add_argument("--download", action="store_true",
                        help="Download dataset from Google Drive before processing.")
    parser.add_argument("--drive-id", type=str, default=config.DRIVE_FILE_ID,
                        help="Google Drive file ID for the dataset zip.")

    # Hardware
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for GPU inference.")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="Number of DataLoader workers.")
    parser.add_argument("--max-workers", type=int, default=config.MAX_WORKERS,
                        help="Number of threads for I/O tasks.")

    # Hyperparameters
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed for reproducibility.")
    parser.add_argument("--n-trials", type=int, default=config.OPTUNA_N_TRIALS,
                        help="Number of Optuna trials for clustering.")
    parser.add_argument("--k-min", type=int, default=config.OPTUNA_K_MIN,
                        help="Minimum number of clusters.")
    parser.add_argument("--k-max", type=int, default=config.OPTUNA_K_MAX,
                        help="Maximum number of clusters.")
    parser.add_argument("--contamination", type=float, default=config.ISOLATION_CONTAMINATION,
                        help="Isolation Forest contamination rate.")
    parser.add_argument("--rho-max", type=float, default=config.RHO_MAX,
                        help="Maximum background saturation.")
    parser.add_argument("--sim-threshold", type=float, default=config.SIMILARITY_THRESHOLD,
                        help="Minimum cosine similarity for habitat matching.")
    parser.add_argument("--max-retries", type=int, default=config.MAX_RETRIES,
                        help="Max synthesis placement attempts.")
    parser.add_argument("--synthesis-batch-size", type=int, default=config.SYNTHESIS_BATCH_SIZE,
                        help="Batch size specifically for synthesis MiDaS.")
    parser.add_argument("--max-synthesis-images", type=int, default=config.MAX_SYNTHESIS_IMAGES,
                        help="Max number of successfully synthesized images to generate.")

    # Output
    parser.add_argument("--output-name", type=str, default="COD10K-AUG-OURS",
                        help="Name for the final output zip (without .zip).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging.")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.verbose)

    output_dir = args.output_dir
    ensure_dir(output_dir)

    # Derived paths
    embeddings_dir = os.path.join(output_dir, "embeddings")
    partitioned_dir_root = os.path.join(output_dir, "partitioned")
    partitioned_dir = os.path.join(partitioned_dir_root, "CSP_Partitioned_Dataset")
    centroids_path = os.path.join(partitioned_dir_root, "habitat_centroids.npy")
    dictionary_path = os.path.join(output_dir, "latent_affordance_dictionary.json")
    catalog_path = os.path.join(output_dir, "environment_catalog.json")
    synth_dir = os.path.join(output_dir, "synthesized")
    final_output = os.path.join(output_dir, args.output_name)

    total_start = time.time()

    # ====================================================================
    # STEP 1: Semantic Embedding
    # ====================================================================
    print("\n" + "=" * 70)
    print("  STEP 1/6: Semantic Embedding & PCA Compression")
    print("=" * 70)

    drive_id = args.drive_id if args.download else None
    extract_embeddings(
        data_dir=args.data_dir,
        output_dir=embeddings_dir,
        drive_file_id=drive_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ====================================================================
    # STEP 2: Habitat Partitioning
    # ====================================================================
    print("\n" + "=" * 70)
    print("  STEP 2/6: Habitat Partitioning & Dataset Routing")
    print("=" * 70)

    run_partitioning(
        embeddings_dir=embeddings_dir,
        data_dir=args.data_dir,
        output_dir=partitioned_dir_root,
        n_trials=args.n_trials,
        k_min=args.k_min,
        k_max=args.k_max,
        max_workers=args.max_workers,
        seed=args.seed,
    )

    # ====================================================================
    # STEP 3: Latent Affordance Dictionary
    # ====================================================================
    print("\n" + "=" * 70)
    print("  STEP 3/6: Latent Affordance Dictionary")
    print("=" * 70)

    build_dictionary(
        partitioned_dir=partitioned_dir,
        output_path=dictionary_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        contamination=args.contamination,
        seed=args.seed,
    )

    # ====================================================================
    # STEP 4: Background Ingestion
    # ====================================================================
    print("\n" + "=" * 70)
    print("  STEP 4/6: Background Ingestion")
    print("=" * 70)

    ingest_backgrounds(
        partitioned_dir=partitioned_dir,
        centroids_path=centroids_path,
        data_dir=args.data_dir,
        output_path=catalog_path,
        rho_max=args.rho_max,
        sim_threshold=args.sim_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ====================================================================
    # STEP 5: Synthesis Engine
    # ====================================================================
    print("\n" + "=" * 70)
    print("  STEP 5/6: Synthesis Engine")
    print("=" * 70)

    synthesize(
        catalog_path=catalog_path,
        dictionary_path=dictionary_path,
        data_dir=args.data_dir,
        output_dir=synth_dir,
        batch_size=args.synthesis_batch_size,
        num_workers=args.num_workers,
        max_retries=args.max_retries,
        seed=args.seed,
        max_synthesis_images=args.max_synthesis_images,
    )

    # ====================================================================
    # STEP 6: Packaging
    # ====================================================================
    print("\n" + "=" * 70)
    print("  STEP 6/6: Dataset Packaging")
    print("=" * 70)

    zip_path = package_dataset(
        synth_dir=synth_dir,
        original_dir=args.data_dir,
        output_path=final_output,
    )

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  ✅ CSP PIPELINE COMPLETE")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Output:     {zip_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
