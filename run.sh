#!/usr/bin/env bash
set -euo pipefail
# AirLLM Server — Linux / macOS Run Script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d venv ]; then
    echo "[ERROR] venv/ not found. Run install.sh first."
    exit 1
fi

source venv/bin/activate

echo ""
echo "============================================================"
echo "  AirLLM Server"
echo "============================================================"
echo ""
echo "  Model:   ${AIRLLM_MODEL:-Qwen/Qwen3.8-27B}"
echo "  Host:    ${AIRLLM_HOST:-0.0.0.0}"
echo "  Port:    ${AIRLLM_PORT:-8000}"
echo "  Context: ${AIRLLM_MAX_CONTEXT:-65536}"
echo "  Device:  ${AIRLLM_DEVICE:-cpu}"
echo ""

python server.py