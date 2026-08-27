@echo off
REM ============================================================================
REM AirLLM OpenAI-Compatible Server — Windows Install Script
REM ============================================================================
REM Creates a Python virtual environment and installs all dependencies.
REM PyTorch is installed in CPU mode (small, fast, always works).
REM The server auto-detects CUDA at startup — if you have an NVIDIA GPU,
REM you can upgrade PyTorch after install with:
REM   venv\Scripts\activate && pip install torch --index-url https://download.pytorch.org/whl/cu124
REM
REM Usage:
REM   install.bat                  — install deps only
REM   install.bat --download-model — install deps AND pre-download model (~16 GB)
REM ============================================================================
setlocal enabledelayedexpansion

title AirLLM Server Installer

echo.
echo ============================================================
echo   AirLLM Server Installer for Windows
echo ============================================================
echo.

REM ---- Check Python ----
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.10+ required
    python --version
    pause
    exit /b 1
)
echo [OK] Python
python --version
echo.

REM ---- Parse flags ----
set DOWNLOAD_MODEL=0
if /I "%~1"=="--download-model" set DOWNLOAD_MODEL=1
if /I "%~2"=="--download-model" set DOWNLOAD_MODEL=1

REM ---- Create venv ----
if not exist venv\ (
    echo [..] Creating virtual environment...
    python -m venv venv
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

echo.
echo [..] Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM ---- Step 1: Install PyTorch (CPU edition) ----
echo.
echo [..] Step 1/3: Installing PyTorch (CPU edition)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if !ERRORLEVEL! neq 0 (
    echo [WARN] CPU PyTorch install failed — trying default index...
    python -m pip install torch
)
echo [OK] PyTorch installed

REM ---- Step 2: Install AirLLM and server deps ----
echo.
echo [..] Step 2/3: Installing AirLLM and server dependencies...
python -m pip install -r requirements.txt
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Dependency install failed
    pause
    exit /b 1
)
echo [OK] All dependencies installed

REM ---- Verify ----
echo.
echo [..] Verifying...
python -c "import torch; print('  PyTorch: '+torch.__version__); print('  CUDA built-in: '+str(torch.backends.cuda.is_built())); print('  CUDA available: '+str(torch.cuda.is_available()))"

REM ---- Step 3: Optional pre-download model ----
if "!DOWNLOAD_MODEL!"=="1" (
    echo.
    echo [..] Step 3/3: Pre-downloading model Qwen/Qwen3.8-27B...
    echo     This downloads ~16 GB from HuggingFace — may take 10-30 min.
    echo.
    python -c "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; from airllm import AutoModel; print('Downloading...'); m=AutoModel.from_pretrained('Qwen/Qwen3.8-27B', device='cpu'); print('Done.')"
    if !ERRORLEVEL! neq 0 (
        echo [WARN] Download had issues. Try: download_model.bat
    ) else (
        echo [OK] Model downloaded and sharded
    )
) else (
    echo.
    echo [..] Step 3/3: Skipped (use --download-model to pre-download)
)

REM ---- Done ----
echo.
echo ============================================================
echo   Install complete!
echo ============================================================
echo.
echo   Start:  run.bat
echo.
echo   The server listens on http://0.0.0.0:8000 (LAN accessible).
echo.
echo   Optional — enable GPU acceleration (RTX 3050 etc.):
echo     venv\Scripts\activate
echo     pip install torch --index-url https://download.pytorch.org/whl/cu124
echo.
echo   Config (set before running):
echo     set AIRLLM_MODEL=Qwen/Qwen3.8-27B
echo     set AIRLLM_PORT=8000
echo.
pause