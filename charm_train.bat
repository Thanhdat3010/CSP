@echo off
REM CHARM Dataset Training Script
REM Usage: charm_train.bat

cd /d "%~dp0..\.."

echo.
echo ======================================
echo Training on CHARM Augmented Dataset
echo ======================================
echo.

REM Default parameters
set DATASET=data/COD10K-CHARM
set EPOCHS=100
set BATCH=64
set LR=0.0001
set IMGSZ=640

REM Allow command line overrides
if not "%1"=="" set DATASET=%1
if not "%2"=="" set EPOCHS=%2
if not "%3"=="" set BATCH=%3
if not "%4"=="" set LR=%4
if not "%5"=="" set IMGSZ=%5

echo Parameters:
echo   Dataset:  %DATASET%/data.yaml
echo   Epochs:   %EPOCHS%
echo   Batch:    %BATCH%
echo   LR:       %LR%
echo   ImgSize:  %IMGSZ%
echo.

python -m yolo detect train ^
    model=yolov8s.pt ^
    data=%DATASET%/data.yaml ^
    epochs=%EPOCHS% ^
    imgsz=%IMGSZ% ^
    batch=%BATCH% ^
    optimizer=AdamW ^
    lr0=%LR% ^
    deterministic=True ^
    seed=42 ^
    mosaic=0.0 ^
    mixup=0.0 ^
    copy_paste=0.0 ^
    project=runs/CHARM ^
    name=charm_v1

echo.
echo ✅ Training complete!
echo Check runs/CHARM/charm_v1/ for results
echo.
