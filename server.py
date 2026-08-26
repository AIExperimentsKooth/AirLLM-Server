"""
AirLLM OpenAI-Compatible API Server
====================================
Serves any AirLLM-supported model (Qwen3.8-27B by default) as an OpenAI-compatible
endpoint over LAN. Works on Windows with CPU or CUDA. Loads and unloads model
layers one at a time, so a 27B model uses ~3.3 GB VRAM / ~8 GB RAM.

Usage:
    python server.py
    # or
    set AIRLLM_MODEL=Qwen/Qwen3.8-27B && python server.py
"""

import os
import sys
import json
import time
import logging
from typing import Optional, List, Union, AsyncGenerator

import asyncio

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

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
HF_TOKEN = os.environ.get("HF_TOKEN", None)                # for gated models
LAYER_SHARDS_PATH = os.environ.get("AIRLLM_SHARDS_PATH", None)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AirLLM OpenAI-Compatible API",
    version="1.0.0",
    description="OpenAI-compatible inference endpoint powered by AirLLM. "
                "Streams model layers one at a time to run large models on "
                "low-VRAM hardware.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Model state  (loaded once on startup, shared across requests)
# ---------------------------------------------------------------------------
class ModelState:
    model = None
    tokenizer = None
    device = "cpu"
    loaded = False
    model_name = ""

state = ModelState()


def _resolve_device() -> str:
    """Pick the best available device.

    Checks three things:
      1. Is torch compiled with CUDA at all? (``torch.backends.cuda.is_built()``)
      2. Is a CUDA device visible and functional? (``torch.cuda.is_available()``)
      3. What did the user request via ``AIRLLM_DEVICE``?
    """
    cuda_built = torch.backends.cuda.is_built()
    cuda_available = cuda_built and torch.cuda.is_available()
    device_env = os.environ.get("AIRLLM_DEVICE", "auto")

    if device_env == "cpu":
        return "cpu"

    if device_env == "cuda":
        if not cuda_built:
            logger.warning(
                "AIRLLM_DEVICE=cuda but torch was NOT compiled with CUDA. "
                "Reinstall PyTorch with CUDA support or leave AIRLLM_DEVICE unset."
            )
            return "cpu"
        if not cuda_available:
            logger.warning("CUDA requested but no CUDA device found — falling back to CPU")
            return "cpu"
        return "cuda"

    # auto — pick the best available
    if cuda_available:
        logger.info("CUDA detected — using GPU")
        return "cuda"
    logger.info("No CUDA device or torch without CUDA support — using CPU")
    return "cpu"


# ---------------------------------------------------------------------------
# Model loader  (called once at startup, never on the request path)
# ---------------------------------------------------------------------------
def load_model_blocking():
    """Called during uvicorn startup.  Downloads the model from HF if not cached."""
    # ── Diagnostics ──────────────────────────────────────────────────────
    cuda_built = torch.backends.cuda.is_built()
    cuda_available = cuda_built and torch.cuda.is_available()
    logger.info("PyTorch %s  CUDA built-in=%s  CUDA available=%s",
                torch.__version__, cuda_built, cuda_available)
    if cuda_available:
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── Step 1: blind CUDA if torch can't use it ────────────────────────
    # AirLLM internally tries torch CUDA functions during from_pretrained()
    # even when it doesn't need them.  If torch was compiled without CUDA,
    # those calls crash with "Torch not compiled with CUDA enabled"
    # regardless of CUDA_VISIBLE_DEVICES.  The set-and-retry below catches
    # that and forces a CPU-mode retry.
    if not cuda_built or not cuda_available:
        logger.info("Setting CUDA_VISIBLE_DEVICES='' to prevent AirLLM CUDA probe")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ── Step 2: load ────────────────────────────────────────────────────
    from airllm import AutoModel as AirLLMAutoModel

    logger.info("=" * 60)
    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info(f"Max context:   {MAX_CONTEXT_LENGTH} tokens")
    logger.info(f"Compression:   {COMPRESSION or 'none'}")
    logger.info("=" * 60)
    logger.info("(First load downloads and shards the model — this may take a while)")
    logger.info("")

    t0 = time.time()
    kwargs = {}
    if COMPRESSION:
        kwargs["compression"] = COMPRESSION
    if HF_TOKEN:
        kwargs["hf_token"] = HF_TOKEN
    if LAYER_SHARDS_PATH:
        kwargs["layer_shards_saving_path"] = LAYER_SHARDS_PATH

    try:
        model = AirLLMAutoModel.from_pretrained(MODEL_NAME, **kwargs)
    except RuntimeError as exc:
        exc_text = str(exc).lower()
        if "cuda" in exc_text:
            logger.warning(
                "CUDA-related error during model load (even though torch "
                "appears CUDA-capable).  Retrying with CUDA blinded."
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            # Also force device=None if AirLLM accepts it
            kwargs["device"] = "cpu"
            model = AirLLMAutoModel.from_pretrained(MODEL_NAME, **kwargs)
        else:
            raise

    tokenizer = model.tokenizer

    # Validate / extend context length
    tokenizer.model_max_length = MAX_CONTEXT_LENGTH

    device = _resolve_device()
    elapsed = time.time() - t0

    state.model = model
    state.tokenizer = tokenizer
    state.device = device
    state.loaded = True
    state.model_name = MODEL_NAME

    logger.info(f"Model loaded in {elapsed:.1f}s on device: {device}")
    logger.info(f"Tokenizer max_length set to: {tokenizer.model_max_length}")
    if device == "cpu" and cuda_available:
        logger.warning(
            "CUDA GPU detected but model is running on CPU. "
            "This probably means AirLLM's internal CUDA calls failed. "
            "Set AIRLLM_DEVICE=cuda to force GPU mode."
        )
    logger.info(f"Server listening on http://{HOST}:{PORT}")
    logger.info("")


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    seed: Optional[int] = None

class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": state.model_name or MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "airllm",
                "permission": [],
            }
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok" if state.loaded else "loading",
        "model": state.model_name or MODEL_NAME,
        "device": state.device,
        "max_context": MAX_CONTEXT_LENGTH,
    }


def _build_generate_kwargs(
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    max_new_tokens: int,
    seed: Optional[int] = None,
) -> dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "return_dict_in_generate": True,
        "do_sample": True,
    }

    # If temperature is effectively 0, use greedy
    if temperature is not None and temperature < 0.01:
        kwargs["do_sample"] = False
        kwargs["temperature"] = 1.0  # ignored when do_sample=False
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k

    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty

    if seed is not None:
        kwargs["seed"] = seed

    return kwargs


def _run_inference(input_ids: torch.Tensor, gen_kwargs: dict) -> tuple:
    """
    Run AirLLM generate on the given input_ids.
    Returns (output_text, prompt_token_count, completion_token_count, elapsed_s).
    """
    t0 = time.time()
    with torch.no_grad():
        output = state.model.generate(input_ids, **gen_kwargs)

    elapsed = time.time() - t0

    # output.sequences is [batch, seq_len]; we took batch_size=1
    new_tokens = output.sequences[0][input_ids.shape[1]:]
    response = state.tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response, input_ids.shape[1], len(new_tokens), elapsed


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not state.loaded:
        raise HTTPException(503, "Model not loaded yet.  Please wait for startup to finish.")

    try:
        # 1 – Build prompt via chat template
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = state.tokenizer.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )

        # 2 – Tokenize (truncate to context window)
        inputs = state.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH,
        )
        input_ids = inputs.input_ids
        if state.device == "cuda":
            input_ids = input_ids.cuda()

        # 3 – Build generation arguments
        gen_kwargs = _build_generate_kwargs(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            max_new_tokens=request.max_tokens or 2048,
            seed=request.seed,
        )

        # 4 – Run model
        response_text, prompt_tokens, completion_tokens, elapsed = _run_inference(input_ids, gen_kwargs)

        logger.info(
            f"chat/completions  model={request.model}  "
            f"prompt={prompt_tokens}  generated={completion_tokens}  "
            f"time={elapsed:.1f}s  tok/s={completion_tokens / max(elapsed, 0.01):.1f}"
        )

        # 5 – Streaming or single response
        if request.stream:
            return _stream_chat_chunks(response_text, request.model)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if not state.loaded:
        raise HTTPException(503, "Model not loaded yet.  Please wait for startup to finish.")

    try:
        inputs = state.tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH,
        )
        input_ids = inputs.input_ids
        if state.device == "cuda":
            input_ids = input_ids.cuda()

        gen_kwargs = _build_generate_kwargs(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=request.max_tokens or 2048,
            seed=request.seed,
        )

        response_text, prompt_tokens, completion_tokens, elapsed = _run_inference(input_ids, gen_kwargs)

        logger.info(
            f"completions  model={request.model}  "
            f"prompt={prompt_tokens}  generated={completion_tokens}  "
            f"time={elapsed:.1f}s  tok/s={completion_tokens / max(elapsed, 0.01):.1f}"
        )

        if request.stream:
            return _stream_text_chunks(response_text, request.model)

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": response_text,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    except Exception as e:
        logger.error(f"Completion error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Streaming helpers  (simulated — AirLLM generates all tokens at once, then
# streams them as SSE chunks for OpenAI API compatibility)
# ---------------------------------------------------------------------------
def _stream_chat_chunks(text: str, model_name: str):
    """Yield chat completion chunks word-by-word."""

    async def generate():
        completion_id = f"chatcmpl-{int(time.time())}"
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk_text = word + (" " if i < len(words) - 1 else "")
            yield json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None,
                        }
                    ],
                }
            ) + "\n\n"
            # tiny delay so the client doesn't buffer everything at once
            await asyncio.sleep(0.01)

        # Final chunk signalling [DONE]
        yield json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _stream_text_chunks(text: str, model_name: str):
    """Yield text completion chunks word-by-word."""

    async def generate():
        completion_id = f"cmpl-{int(time.time())}"
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk_text = word + (" " if i < len(words) - 1 else "")
            yield json.dumps(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {"index": 0, "text": chunk_text, "finish_reason": None}
                    ],
                }
            ) + "\n\n"
            await asyncio.sleep(0.01)

        yield json.dumps(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
            }
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point — load model then serve
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("AirLLM OpenAI-Compatible Server")
    logger.info(f"Model:   {MODEL_NAME}")
    logger.info(f"Host:    {HOST}")
    logger.info(f"Port:    {PORT}")
    logger.info(f"Context: {MAX_CONTEXT_LENGTH} tokens")
    logger.info("")

    # Load the model before the web server starts accepting requests.
    # On first run this downloads ~16 GB from HuggingFace and shards it.
    load_model_blocking()

    # Fire up uvicorn
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        reload=False,
    )