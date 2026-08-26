@echo off
REM ============================================================================
REM AirLLM Server — Windows Run Script
REM ============================================================================
REM Activates the venv and starts the server.
REM Override defaults by setting environment variables before running, e.g.:
REM   set AIRLLM_PORT=8080
REM   set AIRLLM_MODEL=Qwen/Qwen3.8-27B
REM
REM Then double-click run.bat or run it from cmd.
REM ============================================================================
setlocal enabledelayedexpansion

title AirLLM Server

echo.
echo ============================================================
echo   AirLLM OpenAI-Compatible Server
echo ============================================================
echo.

if not exist venv\ (
    echo [ERROR] Virtual environment not found.
    echo         Run install.bat first.
    pause
    exit /b 1
)

REM ---- Activate ----
call venv\Scripts\activate.bat

REM ---- Verify server.py exists ----
if not exist server.py (
    echo [ERROR] server.py not found in current directory.
    pause
    exit /b 1
)

REM ---- Show configuration ----
echo Configuration:
echo   Model:   %AIRLLM_MODEL%        (default: Qwen/Qwen3.8-27B)
echo   Host:    %AIRLLM_HOST%         (default: 0.0.0.0)
echo   Port:    %AIRLLM_PORT%         (default: 8000)
echo   Context: %AIRLLM_MAX_CONTEXT%  (default: 65536)
echo.

REM ---- Start the server ----
echo [..] Starting server (first load downloads ~16 GB model)...
echo.
python server.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server exited with code %ERRORLEVEL%.
    pause
)

REM ---- Deactivate on exit ----
call deactivate 2>nul