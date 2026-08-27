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

REM ---- Handle --force-cpu flag ----
set FORCE_CPU=0
if /I "%~1"=="--force-cpu" (
    set FORCE_CPU=1
    echo.
    echo [INFO] --force-cpu: installing CPU-only PyTorch regardless.
)
if /I "%~2"=="--force-cpu" set FORCE_CPU=1
if /I "%~3"=="--force-cpu" set FORCE_CPU=1

REM ============================================================================
REM Step 1: Install CPU-only PyTorch first (small, fast download)
REM Then use Python to detect if CUDA is actually available.
REM This avoids all the encoding/batch-parsing nonsense with wmic/findstr.
REM ============================================================================
echo.
echo [..] Step 1/4: Installing base PyTorch (CPU edition)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if %ERRORLEVEL% neq 0 (
    echo [WARN] CPU PyTorch install had issues — trying default index...
    python -m pip install torch
)
echo [OK] Base PyTorch installed.

REM ---- Detect CUDA via Python (single-line to avoid cmd.exe escaping issues) ----
echo.
echo [..] Checking for CUDA-capable GPU...
python -c "import torch,os; print('CUDA_AVAILABLE=1' if torch.cuda.is_available() else 'CUDA_AVAILABLE=0')" >"%TEMP%\airllm_cuda.txt"
type "%TEMP%\airllm_cuda.txt"
echo.

REM Parse detection result
findstr "CUDA_AVAILABLE=1" "%TEMP%\airllm_cuda.txt" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    if "!FORCE_CPU!"=="0" (
        set HAS_CUDA=1
        echo [OK] CUDA-capable GPU detected!
    ) else (
        set HAS_CUDA=0
        echo [INFO] --force-cpu active, using CPU-only PyTorch.
    )
) else (
    set HAS_CUDA=0
    echo [INFO] No CUDA-capable GPU detected — using CPU-only PyTorch.
)

REM ---- Step 2: Reinstall PyTorch with CUDA if detected ----
if "!HAS_CUDA!"=="1" (
    echo.
    echo [..] Step 2/4: Installing PyTorch with CUDA support...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if !ERRORLEVEL! neq 0 (
        echo [WARN] CUDA 12.x PyTorch failed. Trying CUDA 11.x...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
    if !ERRORLEVEL! neq 0 (
        echo [WARN] All CUDA PyTorch installs failed. Falling back to CPU...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
    echo [OK] PyTorch reinstalled.
) else (
    echo.
    echo [..] Step 2/4: Skipped (CPU-only torch already installed).
)

REM ---- Step 3: Install AirLLM and server dependencies ----
echo.
echo [..] Step 3/4: Installing AirLLM and server dependencies...
python -m pip install -r requirements.txt
if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Dependency install failed. See error above.
    pause
    exit /b 1
)
echo [OK] All dependencies installed.

REM ---- Verify torch CUDA status ----
echo.
echo [..] Verifying final PyTorch setup...
python -c "import torch; v=torch.__version__; b=torch.backends.cuda.is_built(); a=torch.cuda.is_available(); print('  PyTorch: '+v); print('  CUDA built-in: '+str(b)); print('  CUDA available: '+str(a)); print('  GPU: '+torch.cuda.get_device_name(0)+'  Mem: %.1f GB'%(torch.cuda.get_device_properties(0).total_mem/1e9)) if a else None"

REM ---- Step 4: Optional pre-download model ----
if /I "%~1"=="--download-model" (
    echo.
    echo [..] Step 4/4: Pre-downloading model Qwen/Qwen3.8-27B...
    echo     This will download ~16 GB from HuggingFace and shard it.
    echo     May take 10-30 minutes depending on your internet speed.
    echo.
    python -c "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; from airllm import AutoModel; print('Downloading...'); m=AutoModel.from_pretrained('Qwen/Qwen3.8-27B', device='cpu'); print('Done.')"
    if !ERRORLEVEL! neq 0 (
        echo [WARN] Model download had issues. Retry: download_model.bat
    ) else (
        echo [OK] Model downloaded and sharded.
    )
) else (
    echo.
    echo [..] Step 4/4: Skipped (use --download-model flag to pre-download)
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
echo   Usage:
echo     run.bat              — starts the server (auto-detects CUDA)
echo     run.bat --force-cpu  — force CPU mode even with CUDA torch
echo.
echo   The server listens on http://0.0.0.0:8000 (LAN accessible).
echo.
echo   Configuration (set before running):
echo     set AIRLLM_MODEL=Qwen/Qwen3.8-27B
echo     set AIRLLM_PORT=8000
echo.
pause