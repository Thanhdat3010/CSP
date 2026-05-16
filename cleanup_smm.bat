@echo off
REM Remove all SMM (Semantic Manifold Morphing) files and references
REM Keep only CHARM (Cluster-Aware Hard Example Mining)

cd /d "%~dp0"

echo.
echo ======================================
echo Cleaning up SMM files...
echo ======================================
echo.

REM Remove SMM proposal documents
if exist "docs\SMM_Methodology_Proposal.md" (
    del "docs\SMM_Methodology_Proposal.md"
    echo ✅ Deleted: docs\SMM_Methodology_Proposal.md
)

if exist "docs\SMM_Full_Proposal.md" (
    del "docs\SMM_Full_Proposal.md"
    echo ✅ Deleted: docs\SMM_Full_Proposal.md
)

if exist "docs\SMM_Updated_Proposal.md" (
    del "docs\SMM_Updated_Proposal.md"
    echo ✅ Deleted: docs\SMM_Updated_Proposal.md
)

REM Remove SMM augmentation folder
if exist "src\augmentations\smm" (
    rmdir /s /q "src\augmentations\smm"
    echo ✅ Deleted: src\augmentations\smm
)

REM Remove SMM scripts
if exist "scripts\verify_smm.py" (
    del "scripts\verify_smm.py"
    echo ✅ Deleted: scripts\verify_smm.py
)

if exist "scripts\smm_sample_configs.py" (
    del "scripts\smm_sample_configs.py"
    echo ✅ Deleted: scripts\smm_sample_configs.py
)

if exist "scripts\verify_smm_v2.py" (
    del "scripts\verify_smm_v2.py"
    echo ✅ Deleted: scripts\verify_smm_v2.py
)

REM Remove SMM datasets
if exist "data\COD10K-SMM" (
    rmdir /s /q "data\COD10K-SMM"
    echo ✅ Deleted: data\COD10K-SMM
)

if exist "data\COD10K-SMM-v2" (
    rmdir /s /q "data\COD10K-SMM-v2"
    echo ✅ Deleted: data\COD10K-SMM-v2
)

echo.
echo ======================================
echo ✅ SMM cleanup complete!
echo Now using CHARM only
echo ======================================
echo.
echo Next steps:
echo  1. charm_generate.bat
echo  2. charm_train.bat
echo.
