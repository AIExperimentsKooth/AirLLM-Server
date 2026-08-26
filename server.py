"""
AirLLM OpenAI-Compatible API Server
====================================
Serves any AirLLM-supported model as an OpenAI-compatible endpoint.
Works on any platform (Windows, Linux, macOS) with CPU or CUDA.

Loads and unloads model layers one at a time, so a 27B model uses
~3.3 GB VRAM or ~6-8 GB RAM — no quantization required.

Usage:
    python server.py
    AIRLLM_MODEL=Qwen/Qwen3-32B python server.py
    AIRLLM_DEVICE=cuda python server.py     # enable GPU
"""

import os, sys, json, time, logging, asyncio
from typing import Optional, List, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("airllm-server")

# ---------------------------------------------------------------------------
# Configuration  (all overridable via environment variables)
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("AIRLLM_MODEL", "Qwen/Qwen3.8-27B")
HOST = os.environ.get("AIRLLM_HOST", "0.0.0.0")
PORT = int(os.environ.get("AIRLLM_PORT", "8000"))
MAX_CONTEXT_LENGTH = int(os.environ.get("AIRLLM_MAX_CONTEXT", "65536"))
COMPRESSION = os.environ.get("AIRLLM_COMPRESSION", None)  # "4bit", "8bit", or None
HF_TOKEN = os.environ.get("HF_TOKEN", None)
LAYER_SHARDS_PATH = os.environ.get("AIRLLM_SHARDS_PATH", None)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AirLLM API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ModelState:
    model = None
    tokenizer = None
    device = "cpu"
    loaded = False
    model_name = ""

state = ModelState()


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device() -> str:
    """Return 'cpu' or 'cuda' based on AIRLLM_DEVICE and torch capabilities."""
    cuda_built = torch.backends.cuda.is_built()
    cuda_avail = cuda_built and torch.cuda.is_available()
    env = os.environ.get("AIRLLM_DEVICE", "cpu")  # safe default

    if env == "cpu":
        return "cpu"
    if env == "cuda":
        if not cuda_built:
            logger.warning("AIRLLM_DEVICE=cuda but torch lacks CUDA — using CPU")
            return "cpu"
        if not cuda_avail:
            logger.warning("AIRLLM_DEVICE=cuda but no GPU found — using CPU")
            return "cpu"
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return "cuda"
    # auto
    if cuda_avail:
        logger.info("CUDA available. Set AIRLLM_DEVICE=cuda to enable GPU.")
    return "cpu"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model():
    """Download (if needed) and load the model via AirLLM."""
    # Diagnostics
    device = resolve_device()
    cuda_built = torch.backends.cuda.is_built()
    cuda_avail = cuda_built and torch.cuda.is_available()
    logger.info("torch=%s  CUDA built-in=%s  CUDA available=%s",
                torch.__version__, cuda_built, cuda_avail)
    if cuda_avail:
        logger.info("GPU: %s  VRAM: %.1f GB",
                     torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_mem / 1e9)

    # Build kwargs for AirLLM.AutoModel.from_pretrained
    kwargs = {}

    # ── The core fix ─────────────────────────────────────────────────
    # AirLLM's device="cpu" parameter tells it to never attempt CUDA ops
    # during model loading or inference.  This is the official documented
    # CPU path (v2.10.1+).  We use it by default for stability.
    if device == "cpu":
        kwargs["device"] = "cpu"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    if COMPRESSION:
        kwargs["compression"] = COMPRESSION
    if HF_TOKEN:
        kwargs["hf_token"] = HF_TOKEN
    if LAYER_SHARDS_PATH:
        kwargs["layer_shards_saving_path"] = LAYER_SHARDS_PATH

    # Import after env is prepared
    from airllm import AutoModel as AirLLMAutoModel

    logger.info("=" * 55)
    logger.info(f"Model:     {MODEL_NAME}")
    logger.info(f"Context:   {MAX_CONTEXT_LENGTH}")
    logger.info(f"Device:    {kwargs.get('device', 'auto')}")
    logger.info(f"Compress:  {COMPRESSION or 'none'}")
    logger.info("=" * 55)
    logger.info("(First load downloads ~16 GB from HuggingFace — this takes time)")
    logger.info("")

    t0 = time.time()
    try:
        model = AirLLMAutoModel.from_pretrained(MODEL_NAME, **kwargs)
    except RuntimeError as exc:
        if "cuda" in str(exc).lower():
            logger.warning("CUDA error — retrying with device='cpu' and CUDA blinded")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            kwargs["device"] = "cpu"
            model = AirLLMAutoModel.from_pretrained(MODEL_NAME, **kwargs)
        else:
            raise

    tokenizer = model.tokenizer
    tokenizer.model_max_length = MAX_CONTEXT_LENGTH

    elapsed = time.time() - t0
    state.model = model
    state.tokenizer = tokenizer
    state.device = device
    state.loaded = True
    state.model_name = MODEL_NAME

    logger.info(f"Loaded in {elapsed:.1f}s  device={device}")
    logger.info(f"Tokenizer max_length={tokenizer.model_max_length}")
    if device == "cpu" and cuda_avail:
        logger.info("(GPU present — set AIRLLM_DEVICE=cuda to enable)")
    logger.info(f"Listening: http://{HOST}:{PORT}")
    logger.info("")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatMsg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMsg]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    stream: Optional[bool] = False
    seed: Optional[int] = None

class CompletionReq(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": state.model_name or MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "airllm",
        }],
    }

@app.get("/health")
async def health():
    return {
        "status": "ok" if state.loaded else "loading",
        "model": state.model_name or MODEL_NAME,
        "device": state.device,
        "max_context": MAX_CONTEXT_LENGTH,
    }


def _gen_kwargs(req) -> dict:
    kw = {
        "max_new_tokens": req.max_tokens or 2048,
        "use_cache": True,
        "return_dict_in_generate": True,
        "do_sample": True,
    }
    if req.temperature is not None and req.temperature < 0.01:
        kw["do_sample"] = False
    else:
        if req.temperature is not None:
            kw["temperature"] = req.temperature
        if req.top_p is not None:
            kw["top_p"] = req.top_p
        if req.top_k is not None:
            kw["top_k"] = req.top_k
    if req.repetition_penalty is not None:
        kw["repetition_penalty"] = req.repetition_penalty
    if req.seed is not None:
        kw["seed"] = req.seed
    return kw


def _infer(input_ids, gen_kwargs):
    t0 = time.time()
    with torch.no_grad():
        out = state.model.generate(input_ids, **gen_kwargs)
    elapsed = time.time() - t0
    new_toks = out.sequences[0][input_ids.shape[1]:]
    text = state.tokenizer.decode(new_toks, skip_special_tokens=True)
    return text, input_ids.shape[1], len(new_toks), elapsed


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatReq):
    if not state.loaded:
        raise HTTPException(503, "Model still loading (first download may take long)")
    try:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = state.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH)
        kw = _gen_kwargs(req)
        text, pt, ct, elapsed = _infer(inputs.input_ids, kw)
        logger.info(f"chat  prompt={pt} gen={ct}  {elapsed:.1f}s  {ct/max(elapsed,0.01):.1f}tok/s")
        if req.stream:
            return _stream_chat(text, req.model)
        return {
            "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/v1/completions")
async def completions(req: CompletionReq):
    if not state.loaded:
        raise HTTPException(503, "Model still loading")
    try:
        inputs = state.tokenizer(req.prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH)
        kw = _gen_kwargs(req)
        text, pt, ct, elapsed = _infer(inputs.input_ids, kw)
        logger.info(f"completion  prompt={pt} gen={ct}  {elapsed:.1f}s  {ct/max(elapsed,0.01):.1f}tok/s")
        if req.stream:
            return _stream_text(text, req.model)
        return {
            "id": f"cmpl-{int(time.time())}", "object": "text_completion", "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }
    except Exception as e:
        logger.error(f"Completion error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Streaming (simulated word-by-word)
# ---------------------------------------------------------------------------
def _stream_chat(text: str, model_name: str):
    async def gen():
        cid = f"chatcmpl-{int(time.time())}"
        words = text.split(" ")
        for i, w in enumerate(words):
            chunk = w + (" " if i < len(words) - 1 else "")
            yield json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                              "model": model_name, "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}) + "\n\n"
            await asyncio.sleep(0.01)
        yield json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                          "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


def _stream_text(text: str, model_name: str):
    async def gen():
        cid = f"cmpl-{int(time.time())}"
        words = text.split(" ")
        for i, w in enumerate(words):
            chunk = w + (" " if i < len(words) - 1 else "")
            yield json.dumps({"id": cid, "object": "text_completion", "created": int(time.time()),
                              "model": model_name, "choices": [{"index": 0, "text": chunk, "finish_reason": None}]}) + "\n\n"
            await asyncio.sleep(0.01)
        yield json.dumps({"id": cid, "object": "text_completion", "created": int(time.time()),
                          "model": model_name, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("AirLLM OpenAI-Compatible Server")
    logger.info(f"Model:   {MODEL_NAME}")
    logger.info(f"Host:    {HOST}")
    logger.info(f"Port:    {PORT}")
    logger.info(f"Context: {MAX_CONTEXT_LENGTH}")
    logger.info("")
    load_model()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", reload=False)