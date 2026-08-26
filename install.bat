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

REM ============================================================================
REM NVIDIA GPU detection (Windows-friendly)
REM ============================================================================
REM Strategy: try nvidia-smi first, fall back to wmic (always available).
REM Both work without adding anything to PATH.
REM ============================================================================
set HAS_NVIDIA=0
set CUDA_VER=12

REM Method 1: nvidia-smi (driver-installed, may or may not be in PATH)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set HAS_NVIDIA=1
    for /f "tokens=2 delims=." %%a in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader ^| find "."') do (
        set CUDA_MAJOR=%%a
    )
    REM Extract CUDA version from driver: drivers 525+ = CUDA 12, 470-525 = CUDA 11
    if "!CUDA_MAJOR!" GEQ "525" ( set CUDA_VER=12 ) else ( set CUDA_VER=11 )
)

REM Method 2: wmic (always works on Windows)
if "!HAS_NVIDIA!"=="0" (
    wmic path Win32_VideoController get Name >"%TEMP%\airllm_gpu.txt" 2>nul
    findstr /i "NVIDIA" "%TEMP%\airllm_gpu.txt" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set HAS_NVIDIA=1
        echo [INFO] NVIDIA GPU found via WMI.
        type "%TEMP%\airllm_gpu.txt"
    ) else (
        echo [INFO] No NVIDIA GPU detected — will use CPU-only PyTorch.
    )
    del "%TEMP%\airllm_gpu.txt" 2>nul
)

if "!HAS_NVIDIA!"=="1" (
    echo [OK] NVIDIA GPU detected. CUDA version: !CUDA_VER!
    echo.
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>nul
    echo.
)

REM ---- Create virtual environment ----
if not exist venv\ (
    echo.
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

REM ---- Activate and upgrade pip ----
echo.
echo [..] Upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

REM ---- Choose PyTorch version ----
if /I "%~1"=="--force-cpu" (
    echo.
    echo [INFO] --force-cpu: installing CPU-only PyTorch despite NVIDIA GPU.
    set HAS_NVIDIA=0
)

echo.
if "!HAS_NVIDIA!"=="1" (
    if "!CUDA_VER!"=="12" (
        set TORCH_URL=https://download.pytorch.org/whl/cu124
    ) else (
        set TORCH_URL=https://download.pytorch.org/whl/cu121
    )
    echo [..] Installing PyTorch with CUDA support (from !TORCH_URL!)...
    pip install torch torchvision torchaudio --index-url !TORCH_URL! --quiet
    if !ERRORLEVEL! neq 0 (
        echo [WARN] CUDA PyTorch install failed. Trying CPU version instead...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    )
) else (
    echo [..] Installing CPU-only PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    if !ERRORLEVEL! neq 0 (
        pip install torch --quiet
    )
)

REM ---- Verify torch CUDA status ----
echo.
echo [..] Verifying PyTorch CUDA...
python -c "
import torch
print('  PyTorch version:   ', torch.__version__)
print('  CUDA built-in:     ', torch.backends.cuda.is_built())
print('  CUDA available:    ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU:               ', torch.cuda.get_device_name(0))
"
if %ERRORLEVEL% neq 0 (
    echo [WARN] Torch verification command failed.
)

REM ---- Install AirLLM and server dependencies ----
echo.
echo [..] Installing AirLLM and server dependencies...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency install failed. See error above.
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
    python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from airllm import AutoModel
print('Downloading Qwen/Qwen3.8-27B...')
model = AutoModel.from_pretrained('Qwen/Qwen3.8-27B')
print('Model downloaded and sharded successfully.')
"
    if !ERRORLEVEL! neq 0 (
        echo [WARN] Model download had issues. You can retry later by running:
        echo         download_model.bat
    ) else (
        echo [OK] Model downloaded.
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