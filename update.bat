@echo off
REM ============================================================================
REM AirLLM Server — Update Script
REM ============================================================================
REM Pulls the latest version of the server from GitHub and re-applies
REM any local overrides (venv, cached model, local config).
REM
REM Usage:
REM   update.bat          — pull latest code
REM   update.bat --force  — discard local changes and force-pull
REM ============================================================================
setlocal enabledelayedexpansion

title AirLLM Server Updater

echo.
echo ============================================================
echo   AirLLM Server Updater
echo ============================================================
echo.

REM ---- Check if git repo is initialized ----
if not exist .git\ (
    echo [..] This folder is not a git repository yet.
    echo     Re-initializing from remote...
    echo.
    REM Backup venv and model cache so we don't re-download
    if exist venv\ (
        echo [..] Backing up virtual environment...
        rename venv venv_backup
    )
    if exist "%USERPROFILE%\.cache\huggingface\" (
        set BACKUP_CACHE=1
    )

    git init
    git remote add origin https://github.com/AIExperimentsKooth/AirLLM-Server.git
    git fetch origin
    git checkout -t origin/main

    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to clone repository.
        echo         Check your internet connection and try again.
        if exist venv_backup\ ( rename venv_backup venv )
        pause
        exit /b 1
    )

    REM Restore venv
    if exist venv_backup\ (
        echo [..] Restoring virtual environment...
        rmdir /s /q venv 2>nul
        rename venv_backup venv
    )
) else (
    REM Normal update — pull latest
    if /I "%~1"=="--force" (
        echo [..] Force update — discarding local changes...
        git fetch origin
        git reset --hard origin/main
    ) else (
        echo [..] Pulling latest version from GitHub...
        git pull origin main
        if !ERRORLEVEL! neq 0 (
            echo.
            echo [WARN] Pull failed — you may have local changes.
            echo         Commit or stash them, or use:
            echo           update.bat --force
            pause
            exit /b 1
        )
    )
)

echo [OK] Code updated to latest version.

REM ---- Reinstall dependencies if requirements.txt changed ----
if exist venv\ (
    echo.
    echo [..] Checking for dependency changes...
    call venv\Scripts\activate.bat
    python -m pip install -r requirements.txt
    if !ERRORLEVEL! neq 0 (
        echo [WARN] Dependency reinstall had issues.
    ) else (
        echo [OK] Dependencies up to date.
    )
)

REM ---- Done ----
echo.
echo ============================================================
echo   Update complete!
echo ============================================================
echo.
echo   To start the server:
echo     run.bat
echo.
pause