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

REM ---- Handle flags ----
set FLAG_DOWNLOAD=0
set FLAG_FORCE_CPU=0
if /I "%~1"=="--download-model" set FLAG_DOWNLOAD=1
if /I "%~2"=="--download-model" set FLAG_DOWNLOAD=1
if /I "%~3"=="--download-model" set FLAG_DOWNLOAD=1
if /I "%~1"=="--force-cpu" set FLAG_FORCE_CPU=1
if /I "%~2"=="--force-cpu" set FLAG_FORCE_CPU=1
if /I "%~3"=="--force-cpu" set FLAG_FORCE_CPU=1

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

REM ============================================================================
REM Step 1: Install CPU-only PyTorch (small download, always works)
REM ============================================================================
echo.
echo [..] Step 1/4: Installing base PyTorch (CPU edition)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if !ERRORLEVEL! neq 0 (
    echo [WARN] CPU PyTorch install failed — trying default index...
    python -m pip install torch
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Could not install PyTorch at all. Check your internet connection.
        pause
        exit /b 1
    )
)
echo [OK] Base PyTorch installed.

REM ============================================================================
REM Step 2: Detect CUDA via a .py file (avoids all cmd quoting issues)
REM ============================================================================
echo.
echo [..] Checking for CUDA-capable GPU...

REM Write detection script to a temp file
echo import torch, sys, subprocess, os >"%TEMP%\airllm_cuda_check.py"
echo. >>"%TEMP%\airllm_cuda_check.py"
echo # Method 1: torch CUDA check >>"%TEMP%\airllm_cuda_check.py"
echo if torch.cuda.is_available(): >>"%TEMP%\airllm_cuda_check.py"
echo     print("CUDA_AVAILABLE=1") >>"%TEMP%\airllm_cuda_check.py"
echo     print("GPU=" + torch.cuda.get_device_name(0)) >>"%TEMP%\airllm_cuda_check.py"
echo     sys.exit(0) >>"%TEMP%\airllm_cuda_check.py"
echo. >>"%TEMP%\airllm_cuda_check.py"
echo # Method 2: nvidia-smi >>"%TEMP%\airllm_cuda_check.py"
echo try: >>"%TEMP%\airllm_cuda_check.py"
echo     result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], >>"%TEMP%\airllm_cuda_check.py"
echo                           capture_output=True, text=True, timeout=5) >>"%TEMP%\airllm_cuda_check.py"
echo     if result.returncode == 0 and result.stdout.strip(): >>"%TEMP%\airllm_cuda_check.py"
echo         print("CUDA_AVAILABLE=1") >>"%TEMP%\airllm_cuda_check.py"
echo         print("GPU=" + result.stdout.strip()) >>"%TEMP%\airllm_cuda_check.py"
echo         sys.exit(0) >>"%TEMP%\airllm_cuda_check.py"
echo except: >>"%TEMP%\airllm_cuda_check.py"
echo     pass >>"%TEMP%\airllm_cuda_check.py"
echo. >>"%TEMP%\airllm_cuda_check.py"
echo # Method 3: nvcuda.dll check >>"%TEMP%\airllm_cuda_check.py"
echo dll_path = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "nvcuda.dll") >>"%TEMP%\airllm_cuda_check.py"
echo if os.path.exists(dll_path): >>"%TEMP%\airllm_cuda_check.py"
echo     print("CUDA_AVAILABLE=1") >>"%TEMP%\airllm_cuda_check.py"
echo     print("GPU=NVIDIA detected") >>"%TEMP%\airllm_cuda_check.py"
echo     sys.exit(0) >>"%TEMP%\airllm_cuda_check.py"
echo. >>"%TEMP%\airllm_cuda_check.py"
echo print("CUDA_AVAILABLE=0") >>"%TEMP%\airllm_cuda_check.py"

python "%TEMP%\airllm_cuda_check.py" >"%TEMP%\airllm_cuda_result.txt" 2>&1
type "%TEMP%\airllm_cuda_result.txt"
echo.
del "%TEMP%\airllm_cuda_check.py"

REM Parse result
findstr "CUDA_AVAILABLE=1" "%TEMP%\airllm_cuda_result.txt" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set HAS_CUDA=1
) else (
    set HAS_CUDA=0
)

if "!HAS_CUDA!"=="1" if "!FLAG_FORCE_CPU!"=="1" (
    set HAS_CUDA=0
    echo [INFO] --force-cpu active, keeping CPU-only PyTorch.
)

if "!HAS_CUDA!"=="1" (
    echo [OK] CUDA-capable GPU detected!
) else (
    echo [INFO] No CUDA-capable GPU detected — using CPU-only PyTorch.
)

REM ---- Step 2b: Reinstall PyTorch with CUDA if detected ----
if "!HAS_CUDA!"=="1" (
    echo.
    echo [..] Step 2/4: Installing PyTorch with CUDA support...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if !ERRORLEVEL! neq 0 (
        echo [WARN] CUDA 12.x PyTorch failed. Trying CUDA 11.x...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
    if !ERRORLEVEL! neq 0 (
        echo [WARN] All CUDA installations failed. Falling back to CPU...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        if !ERRORLEVEL! neq 0 (
            python -m pip install torch
        )
    ) else (
        echo [OK] CUDA PyTorch installed.
    )
) else (
    echo.
    echo [..] Step 2/4: Skipped (CPU-only torch stays).
)

REM ============================================================================
REM Step 3: Install AirLLM and server dependencies
REM ============================================================================
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

REM ---- Verify torch ----
echo.
echo [..] Verifying final PyTorch setup...
python -c "import torch; v=torch.__version__; print('  PyTorch: '+v); print('  CUDA built-in: '+str(torch.backends.cuda.is_built())); print('  CUDA available: '+str(torch.cuda.is_available()))"
del "%TEMP%\airllm_cuda_result.txt" 2>nul

REM ============================================================================
REM Step 4: Optional pre-download model
REM ============================================================================
if "!FLAG_DOWNLOAD!"=="1" (
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

REM ============================================================================
REM Done
REM ============================================================================
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