# 🧬 CSP: Context-Synchronized Pipeline

> **A physics-aware data augmentation pipeline for camouflaged object detection.**

CSP generates synthetic camouflaged images by semantically partitioning environments, extracting physical object properties (depth, lighting, texture), and synthesizing new training data with full physics validation.

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Step-by-Step Usage](#-step-by-step-usage)
- [Full Pipeline](#-full-pipeline)
- [Configuration](#-configuration)
- [Citation](#-citation)

---

## 🏗️ Architecture

```
CSP Pipeline
│
├── Phase 1A: Semantic Partitioning
│   ├── Step 1: DINOv2 Embedding + PCA Compression
│   └── Step 2: Optuna-optimized Habitat Clustering + File Routing
│
├── Phase 1B: Object Physics Extraction
│   └── Step 3: Latent Affordance Dictionary (MiDaS + DINOv2 + Isolation Forest)
│
└── Phase 2: Context-Synchronized Synthesis
    ├── Step 4: Background Ingestion + Distance Transform
    ├── Step 5: Physics-Validated Synthesis Engine
    └── Step 6: Unified Dataset Packaging
```

## 🔧 Installation

### Option 1: Conda (Recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate csp
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

Run the entire pipeline end-to-end with a single command:

```bash
# With auto-download from Google Drive
python scripts/run_pipeline.py \
    --data-dir ./data/COD10K \
    --output-dir ./outputs \
    --download \
    --seed 42

# With local dataset
python scripts/run_pipeline.py \
    --data-dir ./data/COD10K \
    --output-dir ./outputs \
    --seed 42
```

## 📖 Step-by-Step Usage

### Step 1: Semantic Embedding

```bash
python scripts/run_embedding.py \
    --data-dir ./data/COD10K \
    --output-dir ./outputs/embeddings \
    --download \
    --batch-size 256 \
    --seed 42
```

### Step 2: Habitat Partitioning

```bash
python scripts/run_partitioning.py \
    --embeddings ./outputs/embeddings \
    --data-dir ./data/COD10K \
    --output-dir ./outputs/partitioned \
    --n-trials 150
```

### Step 3: Latent Affordance Dictionary

```bash
python scripts/run_dictionary.py \
    --partitioned-dir ./outputs/partitioned/CSP_Partitioned_Dataset \
    --output ./outputs/latent_affordance_dictionary.json \
    --contamination 0.10
```

### Step 4: Background Ingestion

```bash
python scripts/run_ingestion.py \
    --partitioned-dir ./outputs/partitioned/CSP_Partitioned_Dataset \
    --centroids ./outputs/partitioned/habitat_centroids.npy \
    --data-dir ./data/COD10K \
    --output ./outputs/environment_catalog.json
```

### Step 5: Synthesis

```bash
python scripts/run_synthesis.py \
    --catalog ./outputs/environment_catalog.json \
    --dictionary ./outputs/latent_affordance_dictionary.json \
    --data-dir ./data/COD10K \
    --output-dir ./outputs/synthesized \
    --batch-size 512
```

### Step 6: Packaging

```bash
python scripts/run_packaging.py \
    --synth-dir ./outputs/synthesized \
    --original-dir ./data/COD10K \
    --output ./outputs/COD10K-AUG-OURS
```

## 🔄 Full Pipeline

```bash
python scripts/run_pipeline.py --help
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | (required) | Path to dataset root |
| `--output-dir` | `./outputs` | Output directory |
| `--download` | `false` | Auto-download from Google Drive |
| `--batch-size` | `256` | GPU batch size |
| `--seed` | `42` | Random seed |
| `--n-trials` | `150` | Optuna trials |
| `--max-retries` | `50` | Synthesis placement retries |

## ⚙️ Configuration

All default hyperparameters are defined in [`csp/config.py`](csp/config.py). CLI arguments override these defaults at runtime.

## Project Structure

```
CSP/
├── csp/                    # Core Python package
│   ├── __init__.py         # Version metadata
│   ├── config.py           # Default hyperparameters
│   ├── utils.py            # Seed, device, logging utilities
│   ├── models.py           # DINOv2 & MiDaS loaders
│   ├── datasets.py         # PyTorch Dataset classes
│   ├── physics.py          # Physics functions (SH, normals, LBP, etc.)
│   ├── embedding.py        # Phase 1A: Embedding + PCA
│   ├── partitioning.py     # Phase 1A: Clustering + Routing
│   ├── dictionary.py       # Phase 1B: Affordance Dictionary
│   ├── ingestion.py        # Phase 2: Background Ingestion
│   ├── synthesis.py        # Phase 2: Synthesis Engine
│   └── packaging.py        # Phase 2: Dataset Packager
│
├── scripts/                # CLI entry points
│   ├── run_embedding.py
│   ├── run_partitioning.py
│   ├── run_dictionary.py
│   ├── run_ingestion.py
│   ├── run_synthesis.py
│   ├── run_packaging.py
│   └── run_pipeline.py     # End-to-end runner
│
├── environment.yml         # Conda environment
├── requirements.txt        # pip dependencies
├── Pipeline.md             # Architecture documentation
└── README.md               # This file
```

## 📝 Citation

```bibtex
@article{csp2025,
  title={Context-Synchronized Pipeline for Camouflaged Object Augmentation},
  author={Thanh Dat},
  year={2025}
}
```

## License

This project is for academic research purposes.
