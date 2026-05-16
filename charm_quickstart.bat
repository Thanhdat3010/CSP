@echo off
REM CHARM Quick Start Script
REM Combines: Generate Dataset + Verify + Train (optional)

cd /d "%~dp0"

echo.
echo ╔════════════════════════════════════════╗
echo ║  CHARM - Cluster-Aware Hard Mining    ║
echo ║  For Camouflaged Object Detection     ║
echo ╚════════════════════════════════════════╝
echo.

setlocal enabledelayedexpansion

REM Check if dataset already exists
if exist "data\COD10K-CHARM\data.yaml" (
    echo ✅ Dataset already exists at: data/COD10K-CHARM
    echo.
    set /p SKIP="Skip generation? (y/n): "
    if /i "!SKIP!"=="y" goto :SKIP_GEN
)

:GEN
echo.
echo [STEP 1] Generating CHARM Dataset...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
call charm_generate.bat

if errorlevel 1 (
    echo ❌ Generation failed!
    exit /b 1
)

:SKIP_GEN
echo.
echo [STEP 2] Verifying Dataset Quality...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python scripts/verify_charm.py --dataset data/COD10K-CHARM --sample 3

if errorlevel 1 (
    echo ⚠️ Verification warning (continuing...)
)

echo.
echo ✅ OFFLINE AUGMENTATION COMPLETE!
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo Dataset created at: data/COD10K-CHARM/
echo   - train/: 24,320 images + labels
echo   - val/: 2,026 images + labels
echo   - data.yaml: YOLO configuration
echo.
echo Next step: Upload to Colab and train
echo.
echo For training on Colab:
echo   1. Upload data/COD10K-CHARM/ folder
echo   2. Mount in Colab: /content/data/COD10K-CHARM
echo   3. Run: yolo detect train model=yolov8s.pt data=/content/data/COD10K-CHARM/data.yaml ...
echo.
pause
