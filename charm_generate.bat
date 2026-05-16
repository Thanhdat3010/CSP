@echo off
REM CHARM Augmentation Dataset Generation Script
REM Usage: charm_generate.bat

cd /d "%~dp0..\..\..\"

echo.
echo ======================================
echo CHARM: Cluster-Aware Hard Example Mining
echo ======================================
echo.

REM Default parameters
set PARTITION_DIR=data/COD10K_Partitioning
set OUTPUT_DIR=data/COD10K-CHARM
set MARGIN=10
set SMOOTH_KERNEL=15
set TOP_K_HARD=3
set SPLIT_RATIO=0.8

REM Allow command line overrides
if not "%1"=="" set PARTITION_DIR=%1
if not "%2"=="" set OUTPUT_DIR=%2
if not "%3"=="" set MARGIN=%3
if not "%4"=="" set SMOOTH_KERNEL=%4
if not "%5"=="" set TOP_K_HARD=%5
if not "%6"=="" set SPLIT_RATIO=%6

echo Parameters:
echo   Partition Dir: %PARTITION_DIR%
echo   Output Dir:    %OUTPUT_DIR%
echo   Margin:        %MARGIN% pixels
echo   Smooth Kernel: %SMOOTH_KERNEL%
echo   Top-K Hard:    %TOP_K_HARD%
echo   Split Ratio:   %SPLIT_RATIO%
echo.

python src/augmentations/charm/generate_charm_dataset.py ^
    --partition_dir %PARTITION_DIR% ^
    --output %OUTPUT_DIR% ^
    --margin %MARGIN% ^
    --smooth_kernel %SMOOTH_KERNEL% ^
    --top_k_hard %TOP_K_HARD% ^
    --split_ratio %SPLIT_RATIO%

echo.
echo ✅ Dataset generation complete!
echo.
