# -*- coding: utf-8 -*-
"""
CSP Default Configuration.

All tunable hyperparameters and default paths are centralized here.
CLI arguments override these defaults at runtime.
"""


# ==============================================================================
# Reproducibility
# ==============================================================================
SEED = 42

import sys

# ==============================================================================
# Hardware — tune based on your GPU (defaults optimized for A100)
# ==============================================================================
BATCH_SIZE = 16
NUM_WORKERS = 0 if sys.platform == "win32" else 2  # Set to 0 on Windows to avoid pickling/spawn issues
MAX_WORKERS = 4           # ThreadPoolExecutor workers (CPU/IO-bound tasks) - Reduced for consumer GPU

# ==============================================================================
# Dataset Ingestion
# ==============================================================================
DRIVE_FILE_ID = "1NAfyqXkxYatSkfoBwGRe3CW-rI9YfaIk"

# ==============================================================================
# DINOv2
# ==============================================================================
DINO_MODEL = "dinov2_vits14"
DINO_INPUT_SIZE = 224
DINO_FEATURE_DIM = 384

# PCA variance retention threshold
PCA_VARIANCE_RATIO = 0.95

# ==============================================================================
# Optuna — Habitat Partitioning
# ==============================================================================
OPTUNA_N_TRIALS = 150
OPTUNA_K_MIN = 5
OPTUNA_K_MAX = 40

# ==============================================================================
# Latent Affordance Dictionary
# ==============================================================================
ISOLATION_CONTAMINATION = 0.10

# ==============================================================================
# Background Ingestion
# ==============================================================================
RHO_MAX = 0.5                 # Maximum background saturation
SIMILARITY_THRESHOLD = 0.65   # Minimum cosine similarity for habitat matching

# ==============================================================================
# Synthesis Engine
# ==============================================================================
SYNTHESIS_BATCH_SIZE = 32     # Reduced for consumer GPU (4GB VRAM)
MAX_SYNTHESIS_IMAGES = 20000  # Cap on the number of generated augmented images
MAX_RETRIES = 15
BLUR_THRESHOLD = 50.0
Z_BUFFER_MARGIN = 0.05
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
SURFACE_NORMAL_THRESHOLD = 0.75
SH_SIMILARITY_THRESHOLD = 0.80
COLOR_SHIFT_MAX = 40.0
SOURCE_CACHE_SIZE = 500

# ==============================================================================
# Packaging
# ==============================================================================
PACKAGE_MAX_WORKERS = 2
