# AirLLM OpenAI-Compatible Server

Serve a 27B parameter model (Qwen3.8-27B) on a **single 4 GB GPU, or even on CPU-only hardware** — using AirLLM's layer-by-layer streaming so your VRAM/RAM never holds more than one layer at a time.

Works on **Windows, Linux, and macOS**. Exposes the model as a standard OpenAI-format API so any tool (Open WebUI, LobeChat, Cursor, VS Code, custom scripts) can use it over your LAN.

```
┌─────────────┐    POST /v1/chat/completions    ┌─────────────────┐
│ Open WebUI  │ ───────────────────────────────→ │                 │
│ LobeChat    │    POST /v1/completions          │  AirLLM Server  │
│ curl / SDK  │ ───────────────────────────────→ │  (port 8000)    │
│ VS Code     │    GET  /v1/models               │                 │
└─────────────┘ ←─────────────────────────────── └─────────────────┘
                                                    │
                                                    ▼
                                              Qwen3.8-27B
                                              (loaded layer-by-layer)
```

## Requirements

| Item | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| VRAM | none (CPU mode) | 4 GB GPU |
| Disk | 40 GB free | 60 GB+ SSD |
| Python | 3.10+ | 3.11+ |
| OS | Windows 10/11, Linux, macOS | any |

Model download: ~16 GB, ~30 GB during sharding, ~20 GB cached.

## Quick Start

**Windows:**

```
install.bat
run.bat
```

**Linux / macOS:**

```bash
bash install.sh
bash run.sh
```

Or clone from GitHub:

```bash
git clone https://github.com/AIExperimentsKooth/AirLLM-Server.git
cd AirLLM-Server
bash install.sh    # or install.bat on Windows
bash run.sh
```

## Updating

**Windows:** `update.bat`
**Linux / macOS:** `bash update.sh`

Keeps your virtual environment and cached model intact.

## Files

```
airllm-server/
├── server.py             # The API server (cross-platform)
├── requirements.txt      # Python dependencies (no torch — installed by scripts)
│
├── install.sh            # Linux / macOS installer
├── run.sh                # Linux / macOS runner
├── update.sh             # Linux / macOS updater
│
├── install.bat           # Windows installer
├── run.bat               # Windows runner
├── update.bat            # Windows updater
├── download_model.bat    # Windows pre-download helper
│
├── README.md             # This file
└── .gitignore
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (uses model's chat template) |
| `POST /v1/completions` | Text completions |
| `GET /health` | Server health check |

### Python example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.100:8000/v1",  # your server's LAN IP
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=512,
)

print(response.choices[0].message.content)
```

## Configuration (Environment Variables)

All settings are optional — defaults work out of the box.

| Variable | Default | Description |
|---|---|---|
| `AIRLLM_MODEL` | `Qwen/Qwen3.8-27B` | HuggingFace model ID |
| `AIRLLM_HOST` | `0.0.0.0` | Bind address |
| `AIRLLM_PORT` | `8000` | HTTP port |
| `AIRLLM_MAX_CONTEXT` | `65536` | Maximum context length in tokens |
| `AIRLLM_DEVICE` | `cpu` | `cpu` or `cuda` |
| `AIRLLM_COMPRESSION` | *(none)* | `4bit` or `8bit` quantization |
| `HF_TOKEN` | *(none)* | HuggingFace token for gated models |

**Linux / macOS:**

```bash
export AIRLLM_PORT=8080
export AIRLLM_DEVICE=cuda
bash run.sh
```

**Windows:**

```
set AIRLLM_PORT=8080
set AIRLLM_DEVICE=cuda
run.bat
```

## Performance Notes

- **First load**: Downloads ~16 GB from HuggingFace and shards into per-layer files. 10-30 min depending on internet speed.
- **Subsequent loads**: ~1-2 min (reads sharded layers from disk).
- **Inference speed**: ~1-3 tok/s on CPU (Ryzen 5), ~5-15 tok/s on mid-range GPU. AirLLM trades speed for memory — each layer is loaded, computed, and discarded.
- **Compression**: `AIRLLM_COMPRESSION=4bit` speeds disk I/O by ~3× with minimal accuracy loss (requires bitsandbytes).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA built-in=True, CUDA available=False` | Normal when torch was compiled with CUDA but no GPU is present. The `device="cpu"` parameter handles this. |
| `"Torch not compiled with CUDA enabled"` | Only happens with CPU-only torch builds. Fixed by `device="cpu"` parameter, with a try/except retry as fallback. |
| `"Model not loaded yet"` | First load takes 10-30 min. Console shows progress. |
| Disk full | Model needs ~30 GB during sharding. Use `download_model.bat` or `--download-model` separately first. |
| Slow inference | Expected on CPU. Set `AIRLLM_COMPRESSION=4bit` for ~3× speedup. |
| Port conflict | Set `AIRLLM_PORT=8080` or another free port. |

## How It Works

AirLLM loads model weights layer-by-layer from disk, discarding each layer after computation. This means a 27B parameter model only needs enough RAM/VRAM for a single layer (~3.3 GB) plus overhead, not the full model size (~54 GB).

The server uses AirLLM's `device="cpu"` parameter, which is the official documented CPU inference path (v2.10.1+). This prevents AirLLM from attempting any CUDA operations internally, making it work reliably on CPU-only machines and systems where torch is compiled with CUDA but has no GPU.

## License

Apache 2.0 (matching AirLLM's license).