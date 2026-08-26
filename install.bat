@echo off
REM ============================================================================
REM AirLLM OpenAI-Compatible Server — Windows Install Script
REM ============================================================================
REM Creates a Python virtual environment, installs all dependencies,
REM and optionally pre-downloads the model so the server starts faster.
REM
REM Usage:
REM   install.bat                  — install deps only
REM   install.bat --download-model — install deps AND download the model (~16 GB)
REM   install.bat --force-cpu      — install CPU-only PyTorch even with NVIDIA GPU
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
    echo [ERROR] Python not found. Please install Python 3.10+ from:
    echo         https://www.python.org/downloads/windows/
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.10 or newer required.
    python --version
    pause
    exit /b 1
)

echo [OK] Python:
python --version
echo.

REM ============================================================================
REM NVIDIA GPU detection (Windows-friendly)
REM ============================================================================
REM Uses wmic (built into Windows) as the primary method — nvidia-smi is
REM often not in PATH even when NVIDIA drivers are installed.
REM ============================================================================
set HAS_NVIDIA=0

REM Method 1: wmic Win32_VideoController (always available on Windows)
wmic path Win32_VideoController get Name >"%TEMP%\airllm_gpu.txt" 2>nul
findstr /i "NVIDIA" "%TEMP%\airllm_gpu.txt" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set HAS_NVIDIA=1
    echo [OK] NVIDIA GPU detected.
    type "%TEMP%\airllm_gpu.txt"
)
del "%TEMP%\airllm_gpu.txt" 2>nul

REM Method 2: nvidia-smi as fallback (gives us driver/CUDA version)
if "!HAS_NVIDIA!"=="1" (
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>nul
        for /f "tokens=2 delims=." %%a in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader ^| find "."') do (
            set CUDA_MAJOR=%%a
        )
        REM driver 525+ = CUDA 12, 470-524 = CUDA 11
        if "!CUDA_MAJOR!" GEQ "525" ( set CUDA_VER=12 ) else ( set CUDA_VER=11 )
    ) else (
        REM no nvidia-smi, default to CUDA 12
        set CUDA_VER=12
    )
    echo.
) else (
    echo [INFO] No NVIDIA GPU detected — will use CPU-only PyTorch.
    echo.
)

REM ---- Create virtual environment ----
if not exist venv\ (
    echo [..] Creating virtual environment...
    python -m venv venv
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

REM ---- Activate venv ----
echo.
echo [..] Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.

REM NOTE: We deliberately do NOT upgrade pip here.  On Windows,
REM "python -m pip install --upgrade pip" often fails because pip
REM cannot overwrite its own running process.  The default pip that
REM ships with the venv is sufficient to install everything we need.

REM ---- Handle --force-cpu flag ----
if /I "%~1"=="--force-cpu" (
    echo.
    echo [INFO] --force-cpu: installing CPU-only PyTorch despite NVIDIA GPU.
    set HAS_NVIDIA=0
)

REM ---- Install PyTorch ----
echo.
if "!HAS_NVIDIA!"=="1" (
    if "!CUDA_VER!"=="12" (
        set TORCH_URL=https://download.pytorch.org/whl/cu124
    ) else (
        set TORCH_URL=https://download.pytorch.org/whl/cu121
    )
    echo [..] Installing PyTorch with CUDA %CUDA_VER% support...
    echo     from !TORCH_URL!
    python -m pip install torch torchvision torchaudio --index-url !TORCH_URL!
    if !ERRORLEVEL! neq 0 (
        echo.
        echo [WARN] CUDA PyTorch install failed. Trying CPU version instead...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        if !ERRORLEVEL! neq 0 (
            echo [WARN] CPU PyTorch also failed — trying default...
            python -m pip install torch
        )
    )
) else (
    echo [..] Installing CPU-only PyTorch...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if !ERRORLEVEL! neq 0 (
        echo [WARN] CPU PyTorch install had issues — trying default...
        python -m pip install torch
    )
)

REM ---- Verify torch ----
echo.
echo [..] Verifying PyTorch CUDA...
python -c "import torch; print('  PyTorch: '+torch.__version__); print('  CUDA built-in: '+str(torch.backends.cuda.is_built())); print('  CUDA available: '+str(torch.cuda.is_available())); print('  CUDA version: '+str(torch.version.cuda))" 2>&1
if !ERRORLEVEL! neq 0 (
    echo [WARN] Torch verification had an issue — continuing anyway.
)

REM ---- Install AirLLM and server deps ----
echo.
echo [..] Installing AirLLM and server dependencies...
python -m pip install -r requirements.txt
if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Dependency install failed. See error above.
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)
echo [OK] All dependencies installed.

REM ---- Optional: pre-download model ----
if /I "%~1"=="--download-model" (
    echo.
    echo [..] Pre-downloading model Qwen/Qwen3.8-27B...
    echo     This will download ~16 GB from HuggingFace and shard it.
    echo     This may take 10-30 minutes depending on your internet speed.
    echo.
    python -c "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''; from airllm import AutoModel; print('Downloading...'); m = AutoModel.from_pretrained('Qwen/Qwen3.8-27B'); print('Done.')"
    if !ERRORLEVEL! neq 0 (
        echo.
        echo [WARN] Model download had issues.
        echo         You can retry later: download_model.bat
    ) else (
        echo [OK] Model downloaded and sharded.
    )
)

REM ---- Done ----
echo.
echo ============================================================
echo   Install complete!
echo ============================================================
echo.
echo   To start the server:
echo     run.bat
echo.
echo   Or manually:
echo     venv\Scripts\activate ^&^& python server.py
echo.
echo   The server will listen on http://0.0.0.0:8000
echo   (accessible from any device on your LAN).
echo.
echo   To change the model or port, set environment variables:
echo     set AIRLLM_MODEL=Qwen/Qwen3.8-27B
echo     set AIRLLM_PORT=8000
echo.
pause