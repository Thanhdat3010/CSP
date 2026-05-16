@echo off
REM ════════════════════════════════════════════════════════════════
REM CHARM Dataset Augmentation - Execute Now
REM ════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                                                                ║
echo ║   CHARM: Cluster-Aware Hard Example Mining                    ║
echo ║   Data Augmentation for Camouflaged Object Detection          ║
echo ║                                                                ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Step 1: Count actual data
echo [STEP 1] Counting dataset...
echo ────────────────────────────────────────────────────────────────
python scripts/count_data.py --partition data/COD10K_Partitioning
if errorlevel 1 (
    echo.
    echo ⚠️  Note: Update --partition path if needed
    echo    Current: data/COD10K_Partitioning
    echo.
)

echo.

REM Step 2: Generate CHARM dataset
echo [STEP 2] Generating CHARM augmented dataset...
echo ────────────────────────────────────────────────────────────────
python src/augmentations/charm/generate_charm_dataset.py ^
    --partition_dir data/COD10K_Partitioning ^
    --output data/COD10K-CHARM ^
    --margin 10 ^
    --smooth_kernel 15 ^
    --top_k_hard 3 ^
    --split_ratio 0.8

if errorlevel 1 (
    echo.
    echo ❌ Augmentation failed!
    echo.
    echo Troubleshooting:
    echo   1. Check --partition_dir path (default: data/COD10K_Partitioning)
    echo   2. Check if images exist in that path
    echo   3. Check disk space (need ~5-10GB for output)
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Augmentation complete!

REM Step 3: Check integrity
echo.
echo [STEP 3] Verifying dataset integrity...
echo ────────────────────────────────────────────────────────────────
python scripts/check_dataset.py --dataset data/COD10K-CHARM

echo.

REM Step 4: Visualize CHARM method
echo [STEP 4] Creating CHARM visualization...
echo ────────────────────────────────────────────────────────────────
python scripts/visualize_charm.py ^
    --dataset data/COD10K_Partitioning ^
    --output charm_visualization.jpg ^
    --samples 3

if errorlevel 1 (
    echo ⚠️ Visualization skipped
) else (
    echo ✅ Visualization: charm_visualization.jpg
)

echo.
echo ════════════════════════════════════════════════════════════════
echo ✅ CHARM AUGMENTATION COMPLETE!
echo ════════════════════════════════════════════════════════════════
echo.
echo 📊 Generated Dataset:
echo    Location: data/COD10K-CHARM/
echo    train/images/  - augmented training images
echo    train/labels/  - corresponding labels
echo    val/images/    - validation images
echo    val/labels/    - validation labels
echo    data.yaml      - YOLO configuration
echo.
echo 📋 Next steps:
echo    1. Review charm_visualization.jpg (check quality)
echo    2. Upload data/COD10K-CHARM/ to Colab
echo    3. Train on Colab with YOLOv8
echo.
pause
