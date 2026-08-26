"""
AirLLM OpenAI-Compatible API Server
====================================
Serves any AirLLM-supported model as an OpenAI-compatible endpoint.
Works on any platform (Windows, Linux, macOS) with CPU or CUDA.

Modes:
    python server.py                    # HTTP API server (default)
    python server.py --chat             # interactive CLI chat
    python server.py --benchmark        # performance benchmark

Loads and unloads model layers one at a time, so a 27B model uses
~3.3 GB VRAM or ~6-8 GB RAM — no quantization required.
"""

import os, sys, json, time, logging, asyncio, math
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
DELETE_ORIGINAL = os.environ.get("AIRLLM_DELETE_ORIGINAL", None)  # "true" to save disk space

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
    """Return 'cpu' or 'cuda' based on AIRLLM_DEVICE and torch capabilities.

    Default (auto): use CUDA when available, fall back to CPU.
    Set AIRLLM_DEVICE=cpu to force CPU even if GPU is present.
    """
    cuda_built = torch.backends.cuda.is_built()
    cuda_avail = cuda_built and torch.cuda.is_available()
    env = os.environ.get("AIRLLM_DEVICE", "auto")

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
    # auto — use GPU when available
    if cuda_avail:
        logger.info("CUDA detected — using GPU: %s  VRAM: %.1f GB",
                     torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_mem / 1e9)
        return "cuda"
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

    # Build kwargs for AirLLM.AutoModel.from_pretrained
    kwargs = {}

    # --- Device selection ---
    if device == "cpu":
        kwargs["device"] = "cpu"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # GPU mode: don't set device kwarg — AirLLM defaults to CUDA when
    # torch.cuda.is_available() and no device= is passed

    # --- Auto-compression for large models ---
    # Models >= 7B benefit greatly from 4-bit compression at minimal
    # accuracy cost.  A 27B model goes from ~54 GB to ~13.5 GB.
    if COMPRESSION is None:
        # Estimate model size from name (number before 'B' in model ID)
        import re
        size_match = re.search(r'[.-]?(\d+)B', MODEL_NAME.split('/')[-1])
        if size_match:
            param_b = int(size_match.group(1))
            if param_b >= 7:
                kwargs["compression"] = "4bit"
                logger.info("Auto-enabled 4-bit compression for %dB model", param_b)
    else:
        kwargs["compression"] = COMPRESSION

    if HF_TOKEN:
        kwargs["hf_token"] = HF_TOKEN
    if LAYER_SHARDS_PATH:
        kwargs["layer_shards_saving_path"] = LAYER_SHARDS_PATH
    if DELETE_ORIGINAL and DELETE_ORIGINAL.lower() in ("true", "1", "yes"):
        kwargs["delete_original"] = True

    # Import after env is prepared
    from airllm import AutoModel as AirLLMAutoModel

    logger.info("=" * 55)
    logger.info(f"Model:     {MODEL_NAME}")
    logger.info(f"Context:   {MAX_CONTEXT_LENGTH}")
    logger.info(f"Device:    {device}{' (GPU+RAM hybrid)' if device == 'cuda' else ''}")
    logger.info(f"Compress:  {kwargs.get('compression', 'none')}")
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
            kwargs.pop("compression", None)
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
    if device == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"VRAM used: {vram_used:.2f} GB  (layers streamed one-at-a-time)")
    else:
        logger.info("(Set AIRLLM_DEVICE=cuda to use GPU if available)")


# ---------------------------------------------------------------------------
# Shared inference helpers
# ---------------------------------------------------------------------------
def _build_gen_kwargs(
    temperature=0.7, top_p=0.9, top_k=None,
    repetition_penalty=None, max_new_tokens=2048, seed=None,
) -> dict:
    kw = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "return_dict_in_generate": True,
        "do_sample": True,
    }
    if temperature is not None and temperature < 0.01:
        kw["do_sample"] = False
    else:
        if temperature is not None:
            kw["temperature"] = temperature
        if top_p is not None:
            kw["top_p"] = top_p
        if top_k is not None:
            kw["top_k"] = top_k
    if repetition_penalty is not None:
        kw["repetition_penalty"] = repetition_penalty
    if seed is not None:
        kw["seed"] = seed
    return kw


def infer(input_ids, gen_kwargs):
    """Run inference, return (text, prompt_tokens, completion_tokens, elapsed_s)."""
    # Move input to the correct device
    if state.device == "cuda":
        input_ids = input_ids.cuda()

    t0 = time.time()
    with torch.no_grad():
        out = state.model.generate(input_ids, **gen_kwargs)
    elapsed = time.time() - t0
    new_toks = out.sequences[0][input_ids.shape[1]:]
    text = state.tokenizer.decode(new_toks, skip_special_tokens=True)
    return text, input_ids.shape[1], len(new_toks), elapsed


def build_chat_prompt(messages):
    """Apply chat template to a list of [{'role':..., 'content':...}] dicts."""
    return state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize(text):
    """Tokenize text and return input_ids tensor (on CPU; moved to device in infer())."""
    inputs = state.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH)
    return inputs.input_ids


# ===================================================================
# MODE 1: CLI Chat
# ===================================================================
def run_chat():
    """Interactive CLI chat session."""
    history = []
    print()
    print("=" * 55)
    print(f"  AirLLM Chat — {state.model_name}")
    print(f"  Device: {state.device}")
    print("=" * 55)
    print("  Type your message and press Enter.")
    print("  Commands:  /exit  /clear  /help")
    print()

    while True:
        try:
            user = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user:
            continue
        if user == "/exit":
            print("Goodbye.")
            break
        if user == "/clear":
            history.clear()
            print("[History cleared]")
            continue
        if user == "/help":
            print("  /exit   — quit")
            print("  /clear  — clear conversation history")
            print("  /help   — this message")
            continue

        history.append({"role": "user", "content": user})

        # Build prompt from full history
        prompt = build_chat_prompt(history)
        input_ids = tokenize(prompt)
        kw = _build_gen_kwargs(max_new_tokens=512)

        # Streaming output to terminal
        t0 = time.time()
        with torch.no_grad():
            out = state.model.generate(input_ids, **kw)
        elapsed = time.time() - t0
        new_toks = out.sequences[0][input_ids.shape[1]:]
        text = state.tokenizer.decode(new_toks, skip_special_tokens=True)

        tok_count = len(new_toks)
        rate = tok_count / max(elapsed, 0.001)

        print(f"AI  > {text}", flush=True)
        print(f"      [{tok_count} tok in {elapsed:.1f}s  {rate:.1f} tok/s]")
        print()

        history.append({"role": "assistant", "content": text})


# ===================================================================
# MODE 2: Benchmark
# ===================================================================
BENCHMARK_PROMPTS = [
    "What is 2+2?",                                          # ~5 tok prompt
    "Explain the theory of relativity in one paragraph.",     # ~15 tok prompt
    "Write a short poem about artificial intelligence.",      # ~12 tok prompt
    "Summarize the plot of the movie Inception.",             # ~10 tok prompt
    "What are the three laws of robotics? List them.",        # ~14 tok prompt
]

BENCHMARK_OUTPUT_LENGTHS = [1, 10, 32, 64, 128, 256]


def run_benchmark():
    """Run a performance benchmark and print results."""
    print()
    print("=" * 65)
    print(f"  AirLLM Benchmark — {state.model_name}")
    print(f"  Device: {state.device}")
    print("=" * 65)
    print()

    # Warmup run (discarded)
    logger.info("Warmup...")
    warmup = build_chat_prompt([{"role": "user", "content": "Hello."}])
    warmup_ids = tokenize(warmup)
    with torch.no_grad():
        _ = state.model.generate(warmup_ids, max_new_tokens=1, use_cache=True, return_dict_in_generate=True)
    logger.info("Warmup complete.\n")

    # ── 1) Benchmark: fixed prompt, varying output length ──────────
    print("─" * 65)
    print(f"  VARYING OUTPUT LENGTH (fixed prompt: ~14 tok)")
    print("─" * 65)
    print(f"  {'max_tokens':>12} {'tok_out':>8} {'time':>8} {'tok/s':>8}  {'latency/tok':>11}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*11}")

    prompt = build_chat_prompt([{"role": "user", "content": "What are the three laws of robotics? List them."}])
    base_ids = tokenize(prompt)
    prompt_tok = base_ids.shape[1]

    output_len_results = []
    for target_len in BENCHMARK_OUTPUT_LENGTHS:
        kw = _build_gen_kwargs(max_new_tokens=target_len, temperature=0.7)
        t0 = time.time()
        with torch.no_grad():
            out = state.model.generate(base_ids.clone(), **kw)
        elapsed = time.time() - t0
        n_tok = out.sequences[0][prompt_tok:].shape[0]
        rate = n_tok / max(elapsed, 0.0001) if n_tok > 0 else 0
        lat_per_tok = (elapsed / n_tok * 1000) if n_tok > 0 else 0
        output_len_results.append((target_len, n_tok, elapsed, rate, lat_per_tok))
        print(f"  {target_len:>12} {n_tok:>8} {elapsed:>7.2f}s {rate:>7.1f}  {lat_per_tok:>9.0f}ms")

    # ── 2) Benchmark: varying prompt length, fixed output ──────────
    print()
    print("─" * 65)
    print(f"  VARYING PROMPT LENGTH (fixed: 32 output tokens)")
    print("─" * 65)
    print(f"  {'prompt_tok':>12} {'tok_out':>8} {'time':>8} {'tok/s':>8} {'latency/tok':>11}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*11}")

    # Build prompts of different lengths by repeating a short phrase
    prompt_base = "What is the meaning of life? "  # ~5 tok
    prompt_len_results = []
    for repeat in [1, 5, 20, 50, 100]:
        p_text = prompt_base * repeat
        p_chat = build_chat_prompt([{"role": "user", "content": p_text}])
        p_ids = tokenize(p_chat)
        p_tok = p_ids.shape[1]

        kw = _build_gen_kwargs(max_new_tokens=32, temperature=0.7)
        t0 = time.time()
        with torch.no_grad():
            out = state.model.generate(p_ids.clone(), **kw)
        elapsed = time.time() - t0
        n_tok = out.sequences[0][p_tok:].shape[0]
        rate = n_tok / max(elapsed, 0.0001) if n_tok > 0 else 0
        lat_per_tok = (elapsed / n_tok * 1000) if n_tok > 0 else 0
        prompt_len_results.append((p_tok, n_tok, elapsed, rate, lat_per_tok))
        print(f"  {p_tok:>12} {n_tok:>8} {elapsed:>7.2f}s {rate:>7.1f}  {lat_per_tok:>9.0f}ms")

    # ── 3) Summary ─────────────────────────────────────────────────
    print()
    print("─" * 65)
    print(f"  SUMMARY")
    print("─" * 65)

    # Aggregate stats
    all_rates = [r[3] for r in output_len_results if r[1] > 0]
    all_rates += [r[3] for r in prompt_len_results if r[1] > 0]

    if all_rates:
        avg_rate = sum(all_rates) / len(all_rates)
        peak_rate = max(all_rates)
        min_rate = min(all_rates)
        print(f"  Average:     {avg_rate:>7.1f} tok/s")
        print(f"  Peak:        {peak_rate:>7.1f} tok/s")
        print(f"  Min:         {min_rate:>7.1f} tok/s")

    # Model-level time breakdown
    print()
    print(f"  Model:       {state.model_name}")
    print(f"  Device:      {state.device}")
    if state.device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM used:   {mem:.2f} GB")

    # Estimated time for common tasks
    print()
    print(f"  ESTIMATED TIMES (at {min_rate if all_rates else 0:.1f}–{max(all_rates) if all_rates else 0:.1f} tok/s):")
    print(f"    100 tokens:   {100/max(avg_rate,0.01):>5.0f}s – {100/max(min_rate,0.01):>5.0f}s") if min_rate > 0 else None
    print(f"    500 tokens:   {500/max(avg_rate,0.01):>5.0f}s – {500/max(min_rate,0.01):>5.0f}s") if min_rate > 0 else None
    print(f"  (wider range = more disk I/O overhead per layer)")
    print()


# ===================================================================
# Schemas (HTTP API mode only)
# ===================================================================
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


# ===================================================================
# HTTP API Endpoints
# ===================================================================
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


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatReq):
    if not state.loaded:
        raise HTTPException(503, "Model still loading (first download may take long)")
    try:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = build_chat_prompt(msgs)
        input_ids = tokenize(prompt)
        kw = _build_gen_kwargs(
            temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_new_tokens=req.max_tokens or 2048, seed=req.seed,
        )
        text, pt, ct, elapsed = infer(input_ids, kw)
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
        input_ids = tokenize(req.prompt)
        kw = _build_gen_kwargs(
            temperature=req.temperature, top_p=req.top_p,
            max_new_tokens=req.max_tokens or 2048, seed=req.seed,
        )
        text, pt, ct, elapsed = infer(input_ids, kw)
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


# ===================================================================
# Entry
# ===================================================================
if __name__ == "__main__":
    # Parse mode from first CLI arg (strip --chat, --benchmark, or nothing)
    mode = "server"
    remaining_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a == "--chat":
            mode = "chat"
        elif a == "--benchmark":
            mode = "benchmark"
        elif a == "--help" or a == "-h":
            print(__doc__)
            sys.exit(0)

    sys.argv = [sys.argv[0]] + remaining_args  # clean args for other consumers

    logger.info("AirLLM OpenAI-Compatible Server")
    logger.info(f"Model:   {MODEL_NAME}")
    logger.info(f"Context: {MAX_CONTEXT_LENGTH}")
    logger.info(f"Mode:    {mode}")
    logger.info("")

    load_model()

    if mode == "chat":
        run_chat()
    elif mode == "benchmark":
        run_benchmark()
    else:
        logger.info(f"Listening: http://{HOST}:{PORT}")
        logger.info("")
        uvicorn.run(app, host=HOST, port=PORT, log_level="info", reload=False)