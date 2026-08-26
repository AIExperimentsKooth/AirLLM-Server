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

REM ---- Detect NVIDIA GPU (nvidia-smi) ----
nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set HAS_NVIDIA=1
    echo [OK] NVIDIA GPU detected.
) else (
    set HAS_NVIDIA=0
    echo [INFO] No NVIDIA GPU detected — will use CPU-only PyTorch.
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
echo.
if "!HAS_NVIDIA!"=="1" (
    echo [..] NVIDIA GPU detected. Installing PyTorch with CUDA support...
    echo     If this fails, re-run install.bat and it will try CPU-only instead.
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
    if !ERRORLEVEL! neq 0 (
        echo [WARN] CUDA PyTorch install failed. Falling back to CPU version...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
        if !ERRORLEVEL! neq 0 (
            pip install torch --quiet
        )
    )
) else (
    echo [..] Installing PyTorch (CPU version — safe on all Windows machines)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    if !ERRORLEVEL! neq 0 (
        echo [WARN] PyTorch CPU install had issues — trying default...
        pip install torch --quiet
    )
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

REM ---- Verify torch CUDA status ----
echo.
echo [..] Verifying PyTorch CUDA availability...
python -c "import torch; print('Torch CUDA built-in:', torch.backends.cuda.is_built()); print('Torch CUDA available:', torch.cuda.is_available())"
if %ERRORLEVEL% neq 0 (
    echo [WARN] Torch verification failed — continuing anyway.
)

REM ---- Optional: pre-download model ----
if /I "%~1"=="--download-model" (
    echo.
    echo [..] Pre-downloading model Qwen/Qwen3.8-27B...
    echo     This will download ~16 GB from HuggingFace and shard it.
    echo     This may take 10-30 minutes depending on your internet speed.
    echo.
    python -c "
from airllm import AutoModel
print('Downloading Qwen/Qwen3.8-27B...')
model = AutoModel.from_pretrained('Qwen/Qwen3.8-27B')
print('Model downloaded and sharded successfully.')
"
    if !ERRORLEVEL! neq 0 (
        echo [WARN] Model download had issues. You can retry later by running:
        echo         python -c "from airllm import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3.8-27B')"
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