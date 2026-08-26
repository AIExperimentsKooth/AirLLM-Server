#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# AirLLM Server — Linux / macOS Install Script
# ============================================================================
# Creates a venv and installs everything. Use --download-model to
# pre-download Qwen3.8-27B (~16 GB).
#
# Usage:
#   bash install.sh                    # install deps
#   bash install.sh --download-model   # install + pre-download model
#   bash install.sh --force-cpu        # skip CUDA even if NVIDIA GPU present
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  AirLLM Server Installer"
echo "============================================================"
echo ""

# ---- Check Python ----
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+."
    exit 1
fi

pyver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "[OK] Python $pyver"
else
    echo "[ERROR] Python 3.10+ required (got $pyver)"
    exit 1
fi

# ---- Create venv ----
if [ ! -d venv ]; then
    echo "[..] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] venv created"
else
    echo "[OK] venv exists"
fi

# ---- Activate ----
source venv/bin/activate

TORCH_URL=""

# ---- GPU detection via Python (more reliable than shell checks) ----
FORCE_CPU=0
for arg in "$@"; do
    [ "$arg" = "--force-cpu" ] && FORCE_CPU=1
done

echo "[..] Checking for CUDA-capable GPU..."
CUDA_AVAIL=$(python3 -c "
import torch, subprocess, os
if torch.cuda.is_available():
    print(1)
    import sys; sys.exit(0)
# Check nvcuda.dll/nvidia-ml on Linux
try:
    r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=3)
    if r.returncode == 0: print(1); sys.exit(0)
except: pass
print(0)
")

if [ "$CUDA_AVAIL" = "1" ] && [ "$FORCE_CPU" = "0" ]; then
    echo "[OK] CUDA-capable GPU detected"
    TORCH_URL="--index-url https://download.pytorch.org/whl/cu124"
else
    echo "[INFO] CPU-only mode"
    if [ "$FORCE_CPU" = "1" ]; then
        echo "       (--force-cpu active)"
    fi
fi

# ---- Install PyTorch ----
echo ""
echo "[..] Installing PyTorch..."
pip install torch torchvision torchaudio $TORCH_URL 2>&1 | tail -3
echo "[OK] PyTorch installed"

# ---- Install deps ----
echo ""
echo "[..] Installing AirLLM and server dependencies..."
pip install -r requirements.txt 2>&1 | tail -3
echo "[OK] Dependencies installed"

# ---- Verify ----
echo ""
echo "[..] Verifying PyTorch CUDA..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA built-in: {torch.backends.cuda.is_built()}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# ---- Optional: pre-download model ----
for arg in "$@"; do
if [ "$arg" = "--download-model" ]; then
    echo ""
    echo "[..] Pre-downloading Qwen/Qwen3.8-27B (~16 GB)..."
    python3 -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from airllm import AutoModel
print('Downloading...')
m = AutoModel.from_pretrained('Qwen/Qwen3.8-27B', device='cpu')
print('Done.')
"
    echo "[OK] Model downloaded"
fi
done

# ---- Done ----
echo ""
echo "============================================================"
echo "  Install complete!"
echo "============================================================"
echo ""
echo "  Start:  bash run.sh"
echo "  Or:     source venv/bin/activate && python server.py"
echo ""
echo "  Configuration (set before running):"
echo "    export AIRLLM_MODEL=Qwen/Qwen3.8-27B"
echo "    export AIRLLM_PORT=8000"
echo "    export AIRLLM_DEVICE=cpu     # or 'cuda'"
echo ""
echo "  Server listens on http://0.0.0.0:8000 (LAN accessible)"
echo ""