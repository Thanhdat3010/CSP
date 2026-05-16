@echo off
REM CSP-Diff: Cluster-Guided Diffusion Background Augmentation
REM Requirements: pip install diffusers transformers accelerate torch Pillow opencv-python pyyaml tqdm

cd /d "%~dp0"

echo.
echo ================================================
echo  CSP-Diff: Cluster-Guided Background Regeneration
echo ================================================
echo.

python src/augmentations/cspdiff/generate_cspdiff_dataset.py ^
    --data_dir data/COD10K_Partitioning ^
    --output data/COD10K-CSPDiff ^
    --num_variants 2 ^
    --margin 20 ^
    --strength 0.75 ^
    --ip_scale 0.6 ^
    --seed 42

echo.
echo Done!
pause
