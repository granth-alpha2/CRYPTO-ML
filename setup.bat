@echo off
setlocal enabledelayedexpansion
set KAGGLE_API_TOKEN=KGAT_bab7779ac9b0a0cc59ddb1444485d549
chcp 65001 >nul 2>&1
title CRYPTO ML - Auto Setup

echo.
echo ================================================================
echo   CRYPTO ML  -  Automated Setup
echo   Privacy-Preserving GNN on Bitcoin Transactions
echo ================================================================
echo.

:: ----------------------------------------------------------------
:: STEP 0 - Python check
:: ----------------------------------------------------------------
echo [STEP 0] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: Python not found.
    echo   Download from: https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during install.
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   OK - Python %PYVER%
echo.

:: ----------------------------------------------------------------
:: STEP 1 - Virtual environment
:: ----------------------------------------------------------------
echo [STEP 1] Setting up virtual environment...
if not exist ".venv" (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo   ERROR: Could not create virtual environment.
        pause & exit /b 1
    )
    echo   Created .venv
) else (
    echo   .venv already exists - skipping
)
call .venv\Scripts\activate.bat
echo   Virtual environment active.
echo.

:: ----------------------------------------------------------------
:: STEP 2 - Install dependencies (smart install for any Python version)
:: ----------------------------------------------------------------
echo [STEP 2] Installing Python dependencies...
echo   Auto-detecting compatible versions for Python %PYVER%...
echo.

:: Upgrade pip first
python -m pip install --upgrade pip -q --no-warn-script-location

:: Install core data packages first (always available)
echo   Installing core packages (pandas, numpy, sklearn, yaml)...
pip install pandas numpy scikit-learn pyyaml tqdm networkx matplotlib seaborn -q --no-warn-script-location

:: Install torch (let pip pick the right version for your Python)
echo   Installing PyTorch (latest compatible version)...
pip install "torch>=2.9.0" -q --no-warn-script-location
if %errorlevel% neq 0 (
    echo   Trying torch without version constraint...
    pip install torch -q --no-warn-script-location
    if %errorlevel% neq 0 (
        echo   ERROR: Could not install PyTorch.
        echo   Try manually: pip install torch
        pause & exit /b 1
    )
)

:: Install torch-geometric (graph neural network library)
echo   Installing PyTorch Geometric...
pip install "torch-geometric>=2.5.3" -q --no-warn-script-location
if %errorlevel% neq 0 (
    pip install torch-geometric -q --no-warn-script-location
)

:: Kaggle, opendatasets and Web Dashboard tools
echo   Installing dataset download and Web UI tools...
pip install kaggle opendatasets jupyter ipykernel fastapi uvicorn -q --no-warn-script-location

echo.
echo   Verifying installation...
python -c "import torch; print('   torch', torch.__version__, 'OK')" 2>&1
python -c "import torch_geometric; print('   torch_geometric', torch_geometric.__version__, 'OK')" 2>&1
python -c "import pandas, numpy, sklearn, yaml; print('   core packages OK')" 2>&1
echo.
echo   All dependencies installed.
echo.

:: ----------------------------------------------------------------
:: STEP 3 - Download Elliptic dataset
:: ----------------------------------------------------------------
echo [STEP 3] Checking Elliptic dataset...
echo.

set DATASET_OK=1
if not exist "data\raw\elliptic_txs_features.csv" set DATASET_OK=0
if not exist "data\raw\elliptic_txs_edgelist.csv" set DATASET_OK=0
if not exist "data\raw\elliptic_txs_classes.csv"  set DATASET_OK=0

if %DATASET_OK%==1 (
    echo   Full dataset present in data\raw\ - skipping download.
    goto :run_pipeline
)

:: Check if bundled benchmark dataset is present
set SAMPLE_OK=1
if not exist "data\sample\elliptic_txs_features.csv" set SAMPLE_OK=0
if not exist "data\sample\elliptic_txs_edgelist.csv" set SAMPLE_OK=0
if not exist "data\sample\elliptic_txs_classes.csv"  set SAMPLE_OK=0

if %SAMPLE_OK%==1 (
    echo   Bundled benchmark dataset found in data\sample\ (8,800 nodes, 9,493 edges).
    echo   Skipping download - ready to run immediately!
    goto :run_pipeline
)

echo   Dataset not found. Starting auto-download...
echo.

:: Check for Kaggle token
set KAGGLE_TOKEN=%USERPROFILE%\.kaggle\kaggle.json

if exist "%KAGGLE_TOKEN%" (
    echo   Kaggle token found. Downloading via API...
    python download_dataset.py --mode kaggle-api
    if %errorlevel%==0 goto :dataset_ok
)

:: No token - guide user
echo.
echo   ============================================================
echo     ACTION REQUIRED: Kaggle API Token (takes ~60 seconds)
echo   ============================================================
echo.
echo   To download automatically, get a free Kaggle API token:
echo.
echo   1. A browser will open - sign in to Kaggle (free account)
echo   2. Scroll to the "API" section
echo   3. Click "Create New Token"  (downloads a kaggle.json file)
echo   4. Copy that file to:
echo      %USERPROFILE%\.kaggle\kaggle.json
echo.
echo   (The folder will be created automatically)
echo.

if not exist "%USERPROFILE%\.kaggle" mkdir "%USERPROFILE%\.kaggle"
echo   Opening Kaggle API settings in your browser...
start "" "https://www.kaggle.com/settings/api"
echo.
echo   After pasting kaggle.json to the .kaggle folder,
echo   press any key to continue...
pause >nul

if exist "%KAGGLE_TOKEN%" (
    echo.
    echo   Token found! Downloading now...
    python download_dataset.py --mode kaggle-api
    if %errorlevel%==0 goto :dataset_ok
)

:: Last resort - manual
echo.
echo   ============================================================
echo     MANUAL DOWNLOAD STEPS:
echo   ============================================================
echo   1. Go to:
echo      https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
echo   2. Click Download - extract the zip
echo   3. Place these 3 files in data\raw\:
echo        elliptic_txs_features.csv
echo        elliptic_txs_edgelist.csv
echo        elliptic_txs_classes.csv
echo   4. Re-run this setup.bat
echo   ============================================================
start "" "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
echo.
echo   After placing the files, press any key to try again...
pause >nul

:: Check one more time
set DATASET_OK=1
if not exist "data\raw\elliptic_txs_features.csv" set DATASET_OK=0
if not exist "data\raw\elliptic_txs_edgelist.csv" set DATASET_OK=0
if not exist "data\raw\elliptic_txs_classes.csv"  set DATASET_OK=0
if %DATASET_OK%==0 (
    echo   Files still not found. Please add them and re-run setup.bat.
    pause & exit /b 1
)

:dataset_ok
echo   Dataset ready in data\raw\
echo.

:: ----------------------------------------------------------------
:: STEP 4 - Run the ML pipeline
:: ----------------------------------------------------------------
:run_pipeline
echo [STEP 4] Running the CRYPTO ML pipeline...
echo   Step 4a - Preprocessing (1-2 min)
echo   Step 4b - Training GNN  (3-10 min depending on hardware)
echo   Step 4c - Evaluating all stages
echo   Step 4d - Generating compliance report
echo.

python main.py --skip-encrypt
if %errorlevel% neq 0 (
    echo.
    echo   Pipeline failed. See error above.
    echo   Common fixes:
    echo     - Re-run setup.bat (installs may have been incomplete)
    echo     - Make sure the 3 CSV files are in data\raw\
    pause & exit /b 1
)

:: ----------------------------------------------------------------
:: STEP 5 - Launch Web Dashboard & Results
:: ----------------------------------------------------------------
echo.
echo [STEP 5] Launching Interactive Web Dashboard...
echo.

start run_app.bat

echo.
echo ================================================================
echo   SUCCESS! CRYPTO ML is running.
echo.
echo   Web Dashboard:  http://127.0.0.1:8000
echo   Model Checkpoint: results\model.pth
echo   Evaluation CSV:   results\evaluation_report.csv
echo   Compliance Report: results\compliance_report.html
echo.
echo   To launch the Web UI anytime in 1-click:
echo     Double-click run_app.bat
echo ================================================================
echo.
pause
