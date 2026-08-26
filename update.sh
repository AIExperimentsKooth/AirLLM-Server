#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# AirLLM Server — Update Script (Linux / macOS)
# ============================================================================
# Pulls the latest code from GitHub while preserving venv and model cache.
#
# Usage:
#   bash update.sh           # pull latest
#   bash update.sh --force   # discard local changes
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  AirLLM Server Updater"
echo "============================================================"
echo ""

if [ ! -d .git ]; then
    echo "[..] Not a git repo — cloning from GitHub..."
    BACKUP_VENV=0
    [ -d venv ] && mv venv venv_backup && BACKUP_VENV=1

    git init
    git remote add origin https://github.com/AIExperimentsKooth/AirLLM-Server.git
    git fetch origin
    git checkout -t origin/main

    if [ "$BACKUP_VENV" = "1" ]; then
        echo "[..] Restoring venv..."
        rm -rf venv 2>/dev/null
        mv venv_backup venv
    fi
else
    if [ "${1:-}" = "--force" ]; then
        echo "[..] Force update — discarding local changes..."
        git fetch origin
        git reset --hard origin/main
    else
        echo "[..] Pulling latest..."
        git pull origin main
    fi
fi

echo "[OK] Code updated"

# Reinstall deps if requirements changed
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
    echo "[..] Checking dependencies..."
    pip install -r requirements.txt 2>&1 | tail -3
    echo "[OK] Dependencies up to date"
fi

echo ""
echo "============================================================"
echo "  Update complete!"
echo "============================================================"
echo ""
echo "  Start:  bash run.sh"