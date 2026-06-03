# 🧬 Context-Synchronized Pipeline (CSP) - Master Architecture & Proposal

This document outlines the architecture, flow, and future roadmap of the **Context-Synchronized Pipeline (CSP)** for generating synthetic camouflage datasets. The pipeline is built to systematically extract semantic features, isolate objects, ingest background environments, and synthesize high-fidelity, physically consistent camouflaged images.

---

## 🏗️ Phase 1: Semantic Partitioning & Dictionary Building

The goal of Phase 1 is to understand the visual "rules" of the world. It breaks down the raw dataset, clusters similar environments, and extracts the physical properties of camouflaged objects to create a "Latent Affordance Dictionary."

### Step 1: Environment Setup & Backbone Loading (Cell 1)
* **Action:** Mounts Google Drive, sets a strict deterministic seed (`SEED = 42`), and downloads the raw YOLO format dataset (`COD10K`).
* **Hardware:** Configures execution variables (`BATCH_SIZE`, `NUM_WORKERS`, `MAX_WORKERS`) optimized for an A100 GPU to maximize throughput.
* **AI Models:** Initializes **DINOv2** (`vits14`) for semantic feature extraction.

### Step 2: Global Semantic Embedding (Cell 2)
* **Action:** Passes all "camo" training images through DINOv2 to extract deep semantic features (384 dimensions).
* **Compression:** Normalizes the features and uses Principal Component Analysis (PCA) to compress them into a dense representation (`V_FINAL`) while retaining 95% of the variance.

### Step 3: Optimal Habitat Partitioning (Cell 3)
* **Action:** Uses **Optuna** to find the mathematically optimal number of environmental clusters ("habitats").
* **Mechanism:** Computes a Ward's Linkage Matrix on the PCA features and minimizes the Davies-Bouldin Index.
* **Output:** The dataset is partitioned into distinct habitats. It calculates and saves the `habitat_centroids.npy` to act as anchors for Phase 2.

### Step 4: Dataset Routing (Cell 4)
* **Action:** Physically reorganizes the raw dataset into the 25 calculated habitats.
* **Mechanism:** Uses multi-threading to rapidly copy the `image`, `label`, and `mask` triplets into the `CSP_Partitioned_Dataset` structure on the local SSD before backing it up.

### Step 5: The Latent Affordance Dictionary (Cell 6)
* **Action:** Extracts every single camouflaged object from the dataset and calculates its physical properties.
* **Physics Extraction:**
    * **Topography:** Uses **MiDaS** depth maps to calculate the `surface_normal` the object is sitting on.
    * **Lighting:** Approximates ambient lighting using `Spherical Harmonics`.
    * **Structure:** Calculates adaptive `solidity` based on the object's mask shape.
* **The Isolation Sieve:** Passes the extracted features through an **Isolation Forest** to flag mathematically anomalous objects (e.g., digitally altered or poorly labeled objects).
* **Output:** A massive JSON dictionary (`latent_affordance_dictionary.json`) defining the physical rules for over 3,500 objects.

---

## 🌍 Phase 2: Environment Ingestion & Synthesis

Phase 2 focuses on finding empty spaces in images and intelligently pasting the objects from Phase 1 into those spaces, ensuring they obey the laws of physics and semantics.

### Step 6: Universal Background Ingestion (Cell 7)
* **Action:** Scans the partitioned dataset and external "non-camo" images to find empty spaces.
* **Mechanism:**
    * Reads existing YOLO bounding boxes and draws them onto a mask.
    * Uses a Distance Transform (`cv2.distanceTransform`) to find the largest available circle of pure empty space (`R_avail_norm`).
    * Matches external "non-camo" images to the closest semantic habitat using the `habitat_centroids` from Cell 3 via Cosine Similarity.
* **Output:** Creates the `environment_catalog.json`—a map of safe, empty canvases ready for object insertion.

### Step 7: Master Synthesis Engine (Cell 8)
* **Action:** The core "Physics Engine." It takes objects from the Dictionary (Cell 6) and pastes them into the Backgrounds (Cell 7).
* **Architecture:** Uses a **Task-Centric Dataloader**. It flattens every possible background-to-object combination into a 1D list to prevent CPU stragglers and maximize GPU batching efficiency.
* **Validation Gates:**
    * **Scale:** Ensures the object fits within the background's empty space.
    * **Topography:** Validates the background surface normal matches the object's original normal.
    * **Lighting:** Validates the background Spherical Harmonics closely match the object's original lighting.
    * **Texture:** Uses Local Binary Patterns (LBP) to ensure the background texture matches the object's original texture.
* **Harmonization:**
    * Applies **Z-Buffer Masking** (MiDaS) to hide the object behind foreground elements (like branches).
    * Applies **LAB Color Transfer** to match ambient color tones.
    * Uses **Poisson Blending** (`cv2.seamlessClone`) and alpha-masking to melt the object into the background seamlessly.
* **Ground Truth:** Dynamically calculates new bounding boxes and segmentation masks for the synthesized object.
* **Output:** Generates a vast folder of entirely synthetic, perfectly labeled camouflaged images.

### Step 8: Unified Dataset Packager (Cell 9)
* **Action:** Merges the original dataset and the new synthetic dataset into a final, train-ready format.
* **Mechanism:** Uses high-speed multi-threading to copy and sort all `.jpg` (images), `.txt` (labels), and `.png` (masks) into a standard flat YOLO directory structure (`/train/image`, `/train/label`, `/train/mask`).
* **Output:** Zips the final `COD10K-AUG-OURS.zip` file to Google Drive, ready to be fed directly into a YOLOv8 training script.
