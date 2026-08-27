"""
AirLLM OpenAI-Compatible API Server
====================================
Serves any AirLLM-supported model as an OpenAI-compatible endpoint.
Works on any platform (Windows, Linux, macOS) with CPU or CUDA.

Modes:
    python server.py                    # HTTP API server (default)
    python server.py --chat             # interactive CLI chat
    python server.py --benchmark        # performance benchmark

Architecture: multi-tier layer cache
  GPU VRAM → System RAM → Disk
  Saturautes available memory, spills overflow to next tier.

  On your machine (RTX 3050 4GB + 40GB RAM + Qwen3.8-27B 4-bit):
    GPU tier: ~6-7 layers permanently resident (~500 MB each)
    RAM tier: all ~40 layers cached (~500 MB each = ~20 GB total)
    Disk:     only first inference or cache misses
"""

import os, sys, json, time, logging, asyncio, math, re, gc, ctypes, types
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
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("AIRLLM_MODEL", "Qwen/Qwen3.8-27B")
HOST = os.environ.get("AIRLLM_HOST", "0.0.0.0")
PORT = int(os.environ.get("AIRLLM_PORT", "8000"))
MAX_CONTEXT_LENGTH = int(os.environ.get("AIRLLM_MAX_CONTEXT", "65536"))
COMPRESSION = os.environ.get("AIRLLM_COMPRESSION", None)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
LAYER_SHARDS_PATH = os.environ.get("AIRLLM_SHARDS_PATH", None)
DELETE_ORIGINAL = os.environ.get("AIRLLM_DELETE_ORIGINAL", None)
CACHE_MODE = os.environ.get("AIRLLM_CACHE", "turbo")  # "turbo" or "stream"
VRAM_HEADROOM = float(os.environ.get("AIRLLM_VRAM_HEADROOM", "0.10"))  # 10% for compute
RAM_HEADROOM = float(os.environ.get("AIRLLM_RAM_HEADROOM", "0.05"))   # 5% for OS
LOCAL_MODEL = os.environ.get("AIRLLM_LOCAL_MODEL", None)  # "true" to keep model files local

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AirLLM API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ModelState:
    model = None         # AirLLM model wrapper
    tokenizer = None
    device = "cpu"
    loaded = False
    model_name = ""
    cache_stats = {}

state = ModelState()


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device() -> str:
    """Return 'cpu' or 'cuda'. Defaults to CUDA when available."""
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
        return "cuda"
    if cuda_avail:
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Local model setup (keep all model files on the same drive as the server)
# ---------------------------------------------------------------------------
def setup_local_model():
    """Create local model directory and configure shards path.

    When --local-model / AIRLLM_LOCAL_MODEL=true is set, the sharded
    layer files (which are read on every inference) are stored next to
    the server.  This keeps them on the same physical drive, reducing
    fragmentation and disk-head seek time.

    Directory layout:
      ./model/splitted/   -- sharded layer files (read on every inference)
      ./model/hf_cache/   -- optional: HuggingFace cache symlink target
      ./model/downloads/  -- optional: HF cached downloads
    """
    import shutil

    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    shards_dir = os.path.join(local, "splitted")
    os.makedirs(shards_dir, exist_ok=True)
    logger.info("Local model directory: %s", local)

    # Point AirLLM's shards here
    global LAYER_SHARDS_PATH
    LAYER_SHARDS_PATH = shards_dir

    # Optionally symlink the HF cache into the local tree so everything
    # lives under one parent directory (no effect on reads, but helps
    # drive-defragmentation and backup tools see a single tree).
    hf_cache_default = os.path.expanduser("~/.cache/huggingface")
    hf_local = os.path.join(local, "hf_cache")
    if os.path.isdir(hf_cache_default) and not os.path.islink(hf_local):
        try:
            os.symlink(hf_cache_default, hf_local)
            logger.info("Symlinked HF cache to %s", hf_local)
        except (OSError, PermissionError):
            # Symlinks may fail on Windows without admin — harmless
            pass

    return local


# ===================================================================
# TurboCache: multi-tier layer caching (GPU → RAM → Disk)
# ===================================================================
class TurboCache:
    """Caches all model layers in RAM, pins as many to GPU as VRAM allows.

    Hooks into AirLLM's layer streaming system by replacing the
    _load_streamed_layer method with a cached version. GPU-pinned
    layers skip eviction (their _post_hook is a no-op), while RAM-only
    layers do a fast CPU→GPU copy instead of disk→GPU.

    On a 40 GB system with an RTX 3050 (4 GB VRAM), Qwen3.8-27B
    with 4-bit compression:
      - GPU pinned:  ~6 layers (~500 MB each ≈ ~3 GB)
      - RAM cached:  all ~40 layers (~20 GB total)
      - Disk:        never touched after first inference
    """

    def __init__(self, airllm, device, vram_headroom=0.10, ram_headroom=0.05):
        self.airllm = airllm
        self.model = airllm.model
        self.device = device
        self.running_device = getattr(airllm, 'running_device', 'cuda:0' if device == 'cuda' else 'cpu')

        self.ram_cache = {}       # {idx: state_dict} on CPU
        self.gpu_pinned = set()   # indices of GPU-pinned layers
        self.gpu_bypass = set()   # idx where weights are pre-loaded on GPU modules
        self.vram_headroom = vram_headroom
        self.ram_headroom = ram_headroom

        # Always-pinned small modules: embed (idx 0) and norm/lm_head (last)
        self.always_pinned = set()

        # Layer count
        self.num_layers = len(airllm.layers)
        self.streamed_indices = list(getattr(airllm, '_streamed_indices', range(self.num_layers)))
        self._original_load = None  # saved original method

    def warmup(self):
        """Pre-load all layers, pin what fits in GPU VRAM."""
        import psutil

        # ---- Identify always-pinned modules ----
        tie = bool(getattr(self.airllm.config, "tie_word_embeddings", False))
        self.always_pinned.add(0)  # embedding
        self.always_pinned.add(self.num_layers - 1)  # norm / lm_head
        if tie:
            self.always_pinned.discard(0)

        # ---- Pre-load ALL streamed layers into RAM ----
        logger.info(f"Pre-loading {len(self.streamed_indices)} layers into RAM...")
        t0 = time.time()
        loaded = 0
        for idx in self.streamed_indices:
            state_dict = self.airllm._load_streamed_layer(idx)
            self.ram_cache[idx] = state_dict
            loaded += 1
        elapsed = time.time() - t0
        logger.info(f"  Loaded {loaded} layers into RAM in {elapsed:.1f}s")

        # ---- Calculate per-layer GPU memory ----
        if not self.ram_cache:
            logger.warning("No layers cached — turbo mode has nothing to work with")
            return

        sample = next(iter(self.ram_cache.values()))
        per_layer_bytes = sum(
            v.numel() * v.element_size() for v in sample.values()
        )
        per_layer_gb = per_layer_bytes / 1e9

        if self.device != "cuda":
            logger.info(f"  CPU mode — all layers cached in RAM only")
            logger.info(f"  RAM used: {len(self.ram_cache) * per_layer_gb:.1f} GB")
            self._install_cached_loader()
            return

        # ---- VRAM budget ----
        vram_total = torch.cuda.get_device_properties(0).total_mem
        vram_budget = int(vram_total * (1.0 - self.vram_headroom))
        logger.info(f"  GPU VRAM: {vram_total/1e9:.1f} GB total, {vram_budget/1e9:.1f} GB budget")

        # Subtract always-pinned
        gpu_budget = vram_budget
        for idx in self.always_pinned:
            if idx in self.ram_cache:
                gpu_budget -= per_layer_bytes

        max_streamed_gpu = max(0, int(gpu_budget // per_layer_bytes))
        logger.info(f"  Layer size: {per_layer_gb:.2f} GB (compressed)")
        logger.info(f"  Fits on GPU: {max_streamed_gpu} streamed layers + {len(self.always_pinned)} pinned")

        # ---- Pin small modules and N streamed layers to GPU ----
        for idx in sorted(self.always_pinned):
            if idx in self.ram_cache:
                self._pin_to_gpu(idx)

        pinned_count = len(self.always_pinned)
        for idx in self.streamed_indices:
            if idx in self.always_pinned:
                continue
            if pinned_count >= (len(self.always_pinned) + max_streamed_gpu):
                break
            self._pin_to_gpu(idx)
            pinned_count += 1

        logger.info(f"  GPU pinned: {pinned_count} layers ({pinned_count * per_layer_gb:.1f} GB)")
        logger.info(f"  RAM cached: {len(self.ram_cache)} layers ({len(self.ram_cache) * per_layer_gb:.1f} GB)")
        ram_total = psutil.virtual_memory().total
        logger.info(f"  RAM free:   {psutil.virtual_memory().available/1e9:.1f} GB / {ram_total/1e9:.1f} GB")

        # ---- Install cached hooks ----
        self._install_cached_loader()

    def _pin_to_gpu(self, idx):
        """Move layer weights to GPU, set on HF model, mark pinned."""
        if idx not in self.ram_cache:
            return
        state_dict = self.ram_cache[idx]
        # Move to GPU and set on model modules
        self.airllm.move_layer_to_device(state_dict)
        self.gpu_pinned.add(idx)
        self.gpu_bypass.add(idx)

    def _install_cached_loader(self):
        """Replace AirLLM's _load_streamed_layer with a cached version.

        For GPU-pinned layers: skip loading entirely (weights already on GPU).
        For RAM-cached layers: return from ram_cache (fast CPU→GPU copy).
        """
        # Save original for any fallback
        orig_load = self.airllm._load_streamed_layer
        orig_post = self.airllm._post_hook
        airllm = self.airllm
        ram_cache = self.ram_cache
        gpu_pinned = self.gpu_pinned
        bypass = self.gpu_bypass
        running_device = self.running_device

        # ---- Cached version of _load_streamed_layer ----
        def cached_load(idx):
            if idx in ram_cache:
                return ram_cache[idx]
            return orig_load(idx)

        # ---- New pre-hook that skips load for pinned layers ----
        def turbo_pre_hook(module, args):
            idx = module._airllm_idx

            if idx in bypass:
                # Weights already on GPU — skip load + move
                module._airllm_moved = []
                # Still trigger prefetch so next layer is ready
                return

            if idx in ram_cache:
                state_dict = ram_cache[idx]
            else:
                state_dict = orig_load(idx)
                ram_cache[idx] = state_dict

            module._airllm_moved = airllm.move_layer_to_device(state_dict)

            # Prefetch next streamed layer into RAM (if not already cached)
            nxt = airllm._next_streamed_idx(idx) if hasattr(airllm, '_next_streamed_idx') else None
            if nxt is not None and nxt not in ram_cache:
                try:
                    ram_cache[nxt] = orig_load(nxt)
                except Exception:
                    pass

        # ---- New post-hook that skips eviction for pinned layers ----
        def turbo_post_hook(module, args, output):
            idx = module._airllm_idx
            if idx in gpu_pinned:
                return output  # Keep weights on GPU

            # Original eviction for non-pinned layers
            if airllm.hf_quantizer is not None or getattr(airllm, '_expert_streaming', False):
                for param_name in getattr(module, '_airllm_moved', []):
                    from accelerate.utils.modeling import set_module_tensor_to_device
                    set_module_tensor_to_device(airllm.model, param_name, 'meta')
            else:
                module.to('meta')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return output

        # ---- Remove old hooks ----
        for idx in range(self.num_layers):
            module = self.airllm.layers[idx]
            module._forward_pre_hooks.clear()
            module._forward_hooks.clear()

        # ---- Register new hooks ----
        for idx in self.streamed_indices:
            module = self.airllm.layers[idx]
            module._airllm_idx = idx
            module.register_forward_pre_hook(turbo_pre_hook)
            module.register_forward_hook(turbo_post_hook)

        # Also patch the generate path for full-model operations
        self.airllm._load_streamed_layer = types.MethodType(cached_load, self.airllm)

    @property
    def stats(self):
        return {
            "mode": "turbo" if self.ram_cache else "stream",
            "layers_ram_cached": len(self.ram_cache),
            "layers_gpu_pinned": len(self.gpu_pinned),
            "total_layers": self.num_layers,
            "device": self.device,
        }


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model():
    """Download (if needed) and load the model via AirLLM."""
    # ── Local model setup (same drive as server) ──
    if LOCAL_MODEL and LOCAL_MODEL.lower() in ("true", "1", "yes", "--local-model"):
        setup_local_model()

    device = resolve_device()
    cuda_built = torch.backends.cuda.is_built()
    cuda_avail = cuda_built and torch.cuda.is_available()
    logger.info("torch=%s  CUDA built-in=%s  CUDA available=%s",
                torch.__version__, cuda_built, cuda_avail)

    kwargs = {}
    if device == "cpu":
        kwargs["device"] = "cpu"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Auto 4-bit compression for models >= 7B
    if COMPRESSION is None:
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

    # ── Monkey-patch torch.cuda before AirLLM import ────────────────
    # AirLLM and bitsandbytes call torch.cuda functions internally even
    # when device="cpu" is passed.  On CPU-only torch builds those calls
    # raise "Torch not compiled with CUDA enabled".  We replace the
    # entire torch.cuda module with a no-op stub so these calls are safe.
    _cuda_built = torch.backends.cuda.is_built()
    _cuda_avail = _cuda_built and torch.cuda.is_available()
    if not _cuda_avail:
        logger.info("Blinding torch.cuda to prevent AirLLM CUDA probe")

        class _CudaStub:
            def is_available(self): return False
            def device_count(self): return 0
            def current_device(self): return 0
            def get_device_name(self, i=0): return "cpu"
            def get_device_properties(self, i=0): return type('obj', (object,), {'total_mem': 1, 'name': 'cpu'})()
            def memory_allocated(self, d=None): return 0
            def max_memory_allocated(self, d=None): return 0
            def memory_reserved(self, d=None): return 0
            def max_memory_reserved(self, d=None): return 0
            def empty_cache(self): return None
            def reset_peak_memory_stats(self, d=None): return None
            def set_device(self, d): return None
            def synchronize(self, d=None): return None
            def stream(self, s=None): return None
            def __getattr__(self, name): return lambda *a, **kw: None

        torch.cuda = _CudaStub()

    from airllm import AutoModel as AirLLMAutoModel

    logger.info("=" * 55)
    logger.info(f"Model:     {MODEL_NAME}")
    logger.info(f"Context:   {MAX_CONTEXT_LENGTH}")
    logger.info(f"Device:    {device}")
    logger.info(f"Compress:  {kwargs.get('compression', 'none')}")
    logger.info(f"Cache:     {CACHE_MODE}")
    if LAYER_SHARDS_PATH:
        logger.info(f"Shards:    {LAYER_SHARDS_PATH}")
    logger.info("=" * 55)
    logger.info("")

    t0 = time.time()
    try:
        model = AirLLMAutoModel.from_pretrained(MODEL_NAME, **kwargs)
    except RuntimeError as exc:
        if "cuda" in str(exc).lower():
            logger.warning("CUDA error — retrying CPU mode")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            kwargs.pop("compression", None)
            kwargs["device"] = "cpu"
            model = AirLLMAutoModel.from_pretrained(MODEL_NAME, **kwargs)
            device = "cpu"
        else:
            raise

    tokenizer = model.tokenizer
    tokenizer.model_max_length = MAX_CONTEXT_LENGTH

    elapsed = time.time() - t0
    state.model = model
    state.tokenizer = tokenizer
    state.device = device
    state.model_name = MODEL_NAME

    logger.info(f"Base model loaded in {elapsed:.1f}s")
    logger.info(f"Tokenizer max_length={tokenizer.model_max_length}")

    # ── TurboCache initialization ──
    if CACHE_MODE == "turbo" and hasattr(model, 'layers') and len(model.layers) > 0:
        logger.info("")
        logger.info("─" * 55)
        logger.info("  Initializing TurboCache (multi-tier layer cache)...")
        logger.info("─" * 55)
        logger.info("")

        cache = TurboCache(
            model, device,
            vram_headroom=VRAM_HEADROOM,
            ram_headroom=RAM_HEADROOM,
        )
        t0 = time.time()
        cache.warmup()
        cache_elapsed = time.time() - t0
        state.cache_stats = cache.stats

        logger.info("")
        logger.info(f"TurboCache ready in {cache_elapsed:.1f}s")
        logger.info(f"  GPU pinned: {cache.stats['layers_gpu_pinned']} layers")
        logger.info(f"  RAM cached: {cache.stats['layers_ram_cached']} layers")
        logger.info("")
    else:
        state.cache_stats = {"mode": "stream", "reason": "layers unavailable or CACHE_MODE=stream"}

    state.loaded = True
    if device == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"VRAM used after cache: {vram_used:.2f} GB")
    logger.info(f"Model ready: {MODEL_NAME}")
    logger.info("")


# ---------------------------------------------------------------------------
# Shared inference helpers
# ---------------------------------------------------------------------------
def _build_gen_kwargs(temperature=0.7, top_p=0.9, top_k=None,
                      repetition_penalty=None, max_new_tokens=2048, seed=None):
    kw = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "return_dict_in_generate": True,
        "do_sample": True,
    }
    if temperature is not None and temperature < 0.01:
        kw["do_sample"] = False
    else:
        if temperature is not None: kw["temperature"] = temperature
        if top_p is not None: kw["top_p"] = top_p
        if top_k is not None: kw["top_k"] = top_k
    if repetition_penalty is not None:
        kw["repetition_penalty"] = repetition_penalty
    if seed is not None:
        kw["seed"] = seed
    return kw


def infer(input_ids, gen_kwargs):
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
    return state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize(text):
    inputs = state.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH)
    return inputs.input_ids


# ===================================================================
# CLI Chat
# ===================================================================
def run_chat():
    history = []
    print()
    print("=" * 55)
    print(f"  AirLLM Chat — {state.model_name}")
    print(f"  Device: {state.device}")
    print(f"  Cache:  {state.cache_stats.get('mode', 'stream')}")
    if state.cache_stats.get('layers_gpu_pinned'):
        print(f"  GPU layers: {state.cache_stats['layers_gpu_pinned']}")
    print("=" * 55)
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
            print("Goodbye."); break
        if user == "/clear":
            history.clear(); print("[History cleared]"); continue
        if user == "/help":
            print("  /exit  /clear  /help"); continue

        history.append({"role": "user", "content": user})
        prompt = build_chat_prompt(history)
        input_ids = tokenize(prompt)
        kw = _build_gen_kwargs(max_new_tokens=512)

        t0 = time.time()
        with torch.no_grad():
            out = state.model.generate(input_ids, **kw)
        elapsed = time.time() - t0
        new_toks = out.sequences[0][input_ids.shape[1]:]
        text = state.tokenizer.decode(new_toks, skip_special_tokens=True)
        rate = len(new_toks) / max(elapsed, 0.001)

        print(f"AI  > {text}", flush=True)
        print(f"      [{len(new_toks)} tok in {elapsed:.1f}s  {rate:.1f} tok/s]")
        print()
        history.append({"role": "assistant", "content": text})


# ===================================================================
# Benchmark
# ===================================================================
BENCHMARK_OUTPUT_LENGTHS = [1, 10, 32, 64, 128, 256]

def run_benchmark():
    print()
    print("=" * 65)
    print(f"  AirLLM Benchmark — {state.model_name}")
    print(f"  Device: {state.device}")
    print(f"  Cache:  {state.cache_stats.get('mode', 'stream')}")
    if state.cache_stats.get('layers_gpu_pinned'):
        print(f"  GPU pinned: {state.cache_stats['layers_gpu_pinned']} layers")
    print("=" * 65)
    print()

    logger.info("Warmup...")
    warmup = build_chat_prompt([{"role": "user", "content": "Hello."}])
    warmup_ids = tokenize(warmup)
    with torch.no_grad():
        _ = state.model.generate(warmup_ids, max_new_tokens=1, use_cache=True, return_dict_in_generate=True)
    logger.info("Warmup complete.\n")

    # ── 1) Varying output length ──
    print("─" * 65)
    print("  VARYING OUTPUT LENGTH (fixed prompt: ~14 tok)")
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

    # ── 2) Varying prompt length ──
    print()
    print("─" * 65)
    print("  VARYING PROMPT LENGTH (fixed: 32 output tokens)")
    print("─" * 65)
    print(f"  {'prompt_tok':>12} {'tok_out':>8} {'time':>8} {'tok/s':>8} {'latency/tok':>11}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*11}")

    prompt_base = "What is the meaning of life? "
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

    # ── Summary ──
    print()
    print("─" * 65)
    print("  SUMMARY")
    print("─" * 65)

    all_rates = [r[3] for r in output_len_results if r[1] > 0]
    all_rates += [r[3] for r in prompt_len_results if r[1] > 0]

    if all_rates:
        avg_rate = sum(all_rates) / len(all_rates)
        peak_rate = max(all_rates)
        min_rate = min(all_rates)
        print(f"  Average:     {avg_rate:>7.1f} tok/s")
        print(f"  Peak:        {peak_rate:>7.1f} tok/s")
        print(f"  Min:         {min_rate:>7.1f} tok/s")

    print(f"  Model:       {state.model_name}")
    print(f"  Device:      {state.device}")
    if state.device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM used:   {mem:.2f} GB")
    print(f"  Cache:       {state.cache_stats.get('mode', 'stream')}")
    if state.cache_stats.get('layers_gpu_pinned'):
        print(f"  GPU pinned:  {state.cache_stats['layers_gpu_pinned']} layers")
    if state.cache_stats.get('layers_ram_cached'):
        print(f"  RAM cached:  {state.cache_stats['layers_ram_cached']} layers")

    if all_rates:
        print(f"")
        print(f"  ESTIMATED TIMES (at {min_rate:.1f}–{peak_rate:.1f} tok/s):")
        print(f"    100 tokens:   {100/max(avg_rate,0.01):>5.0f}s")
        print(f"    500 tokens:   {500/max(avg_rate,0.01):>5.0f}s")
    print()


# ===================================================================
# HTTP API Schemas & Endpoints
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


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": state.model_name or MODEL_NAME, "object": "model",
                   "created": int(time.time()), "owned_by": "airllm"}],
    }

@app.get("/health")
async def health():
    return {
        "status": "ok" if state.loaded else "loading",
        "model": state.model_name or MODEL_NAME,
        "device": state.device,
        "max_context": MAX_CONTEXT_LENGTH,
        "cache": state.cache_stats,
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatReq):
    if not state.loaded:
        raise HTTPException(503, "Model still loading")
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
# Streaming (simulated)
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
    mode = "server"
    for a in sys.argv[1:]:
        if a == "--chat": mode = "chat"
        elif a == "--benchmark": mode = "benchmark"
        elif a == "--local-model": LOCAL_MODEL = "true"
        elif a == "--help" or a == "-h": print(__doc__); sys.exit(0)

    logger.info("AirLLM OpenAI-Compatible Server")
    logger.info(f"Model:   {MODEL_NAME}")
    logger.info(f"Context: {MAX_CONTEXT_LENGTH}")
    logger.info(f"Cache:   {CACHE_MODE}")
    if LOCAL_MODEL and LOCAL_MODEL.lower() in ("true", "1", "yes", "--local-model"):
        logger.info(f"Local:   yes (shards in ./model/)")
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