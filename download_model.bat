@echo off
REM ============================================================================
REM Pre-download the model so the server starts faster on first run.
REM Usage:
REM   download_model.bat              -- download default model (Qwen3.8-27B)
REM   download_model.bat Qwen/Qwen3-32B  -- download a different model
REM ============================================================================
setlocal enabledelayedexpansion

title AirLLM Model Downloader

set MODEL=%~1
if "%MODEL%"=="" set MODEL=Qwen/Qwen3.8-27B

echo.
echo ============================================================
echo   Downloading: %MODEL%
echo ============================================================
echo.
echo This will download ~16 GB from HuggingFace and shard it into
echo per-layer files. This is a one-time operation — subsequent
echo server starts will load from cache.
echo.

if not exist venv\ (
    echo [ERROR] Virtual environment not found.
    echo         Run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [..] Starting download (CUDA blinded for compatibility)...

python -c ^
"import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''; from airllm import AutoModel; print('Downloading %MODEL%...'); model = AutoModel.from_pretrained('%MODEL%'); print('Done!')"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Download failed.
    echo         Check your internet connection and disk space.
    echo         The model needs ~30 GB free for download + sharding.
    pause
    exit /b 1
)

echo.
echo [OK] Model %MODEL% downloaded and sharded successfully.
echo     You can now run the server with:  run.bat
pause