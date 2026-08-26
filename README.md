# AirLLM OpenAI-Compatible Server

Serve a 27B parameter model (Qwen3.8-27B) on a **single 4 GB GPU, or even on CPU** — using AirLLM's layer-by-layer streaming so your VRAM/RAM never holds more than one layer at a time. Exposes the model as a standard OpenAI-format API so any tool (Open WebUI, LobeChat, LibreChat, Cursor, VS Code Copilot, custom scripts) can use it over your LAN.

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
| VRAM | integrated (CPU mode) | 4 GB GPU |
| Disk | 40 GB free | 60 GB+ SSD |
| Python | 3.10+ | 3.11+ |
| OS | Windows 10/11 | Windows 11 |

Model size: ~16 GB download, ~30 GB during sharding (temporary), ~20 GB cached afterward.

## Quick Start

```
install.bat
run.bat
```

Or clone directly from GitHub:

```
git clone https://github.com/AIExperimentsKooth/AirLLM-Server.git
cd AirLLM-Server
install.bat
run.bat
```

## Updating

```
update.bat
```

Pulls the latest version from GitHub. Keeps your virtual environment and cached model intact.

**Step by step:**

1. **Install** — double-click `install.bat`. It creates a Python venv and installs PyTorch (CPU edition, safe on any Windows machine) plus AirLLM and the web server.

2. **Run** — double-click `run.bat`. The **first run downloads ~16 GB** from HuggingFace and shards it into per-layer files. This takes 10-30 minutes depending on your internet. Subsequent starts load from cache in ~1-2 minutes.

3. **Use it** — once the server says "listening on http://0.0.0.0:8000", you can curl it:

```
curl http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"Qwen/Qwen3.8-27B\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
```

## Files

```
airllm-server/
├── install.bat           # One-click setup (venv + deps)
├── run.bat               # Start the server
├── download_model.bat    # Pre-download the model before first run
├── update.bat           # Pull the latest code from GitHub
├── server.py             # The actual API server
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## API Endpoints

The server implements the OpenAI API spec. Use any OpenAI-compatible client:

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
    api_key="not-needed",                       # AirLLM doesn't require a key
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=512,
)

print(response.choices[0].message.content)
```

### Open WebUI / LobeChat

In your chat UI's provider settings:
- **API URL**: `http://<YOUR_LAN_IP>:8000/v1`
- **API Key**: leave blank or enter any string
- **Model**: `Qwen/Qwen3.8-27B`

## Configuration (Environment Variables)

All settings are optional — the defaults work out of the box.

| Variable | Default | Description |
|---|---|---|
| `AIRLLM_MODEL` | `Qwen/Qwen3.8-27B` | HuggingFace model ID |
| `AIRLLM_HOST` | `0.0.0.0` | Bind address (0.0.0.0 = all interfaces) |
| `AIRLLM_PORT` | `8000` | HTTP port |
| `AIRLLM_MAX_CONTEXT` | `65536` | Maximum context length in tokens |
| `AIRLLM_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `AIRLLM_COMPRESSION` | *(none)* | `4bit` or `8bit` for block-wise quantization |
| `HF_TOKEN` | *(none)* | HuggingFace token for gated models |
| `AIRLLM_SHARDS_PATH` | *(default cache)* | Custom path for layer shards |

Set them before starting:

```
set AIRLLM_PORT=8080
set AIRLLM_DEVICE=auto
run.bat
```

## Performance Notes

- **First load**: Downloads ~16 GB from HuggingFace and shards into per-layer files. Takes 10-30 min depending on internet speed.
- **Subsequent loads**: ~1-2 min (reads sharded layers from disk).
- **Inference speed**: ~1-3 tokens/sec on CPU (Ryzen 5), ~5-15 tok/s on a mid-range GPU. AirLLM trades speed for memory efficiency — each layer is loaded, computed, and discarded before the next.
- **Compression**: Set `AIRLLM_COMPRESSION=4bit` to shrink each layer's weight size by 4×, speeding up disk I/O by ~3× with minimal accuracy loss (requires bitsandbytes).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `"Model not loaded yet"` | Wait — first load takes 10-30 min. Check the console for progress. |
| Disk full | The model needs ~30 GB during sharding. Run `download_model.bat` first to see the real size. |
| Slow inference | Expected on CPU. Use `AIRLLM_COMPRESSION=4bit` for ~3× speedup. |
| HuggingFace download fails | Check your internet. Try `download_model.bat` separately. |
| No CUDA GPU | The server auto-detects and falls back to CPU. No action needed. |
| Port conflict | Set `AIRLLM_PORT=8080` or another free port. |

## License

Apache 2.0 (matching AirLLM's license).