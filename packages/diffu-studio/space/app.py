"""Hugging Face ZeroGPU Space entry point for Diffu Studio.

Spaces auto-runs ``app.py``. This reuses diffu_studio's own build components (the SAME path as
``diffu_studio.__main__.main()``) but adapts them to a Space: the single model is pulled from the Hub
(there is no local ``checkpoints/`` dir), style crops ship under ``assets/styles/``, and the ZeroGPU
constraints a local launcher does not have are applied — no forced HF offline, no ``CUDA_VISIBLE_DEVICES``
pin, and the eager-CUDA recognizer is disabled (CUDA must init only inside the forked ``@spaces.GPU`` call).

AoT: on first model load the engine calls ``aot_compile_backbone`` (aokit) — a no-op / fallback-safe off
ZeroGPU, and on ZeroGPU it AoT-compiles the MMDiT backbone for the H200 (falls back to JIT/eager on any
export failure, so the Space always works).

PiD 4× upscaler: ``pid`` is a PINNED GIT DEPENDENCY in requirements.txt (no runtime clone/pip-install).
Its weights are prefetched at boot (CPU-safe) and the model is preloaded in the parent process below, so
the time-budgeted ``@spaces.GPU`` upscale click never pays for a download or a cold build.
"""

from __future__ import annotations

import glob
import os

# 1) Allocator guard BEFORE any torch import (reads env at init). Reuse the studio's helper.
from diffu_studio.device import configure_memory

configure_memory()

# 2) A Space has NO local weight cache — the VAE (Qwen/Qwen-Image), the DINOv3 style encoder and (if used)
#    the recognizer download on first build. __main__ forces HF offline for the local box; a Space must not.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# 3) Register ZeroGPU (import side-effect sets up the fork-based GPU scheduler; SPACES_ZERO_GPU, set by the
#    platform, is what diffu_studio.device.resolve_device() autodetects as the 'zerogpu' backend).
import spaces  # noqa: E402, F401
from diffu_studio.checkpoints import ArchConfig  # noqa: E402
from diffu_studio.device import resolve_device  # noqa: E402
from diffu_studio.engine import StudioEngine  # noqa: E402
from diffu_studio.ui import build_blocks  # noqa: E402
from diffu_studio.upscale import prefetch_weights  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

# PiD weights (decoder + Flux VAE + Gemma-2 text encoder) → HF cache at boot, on CPU time not GPU budget.
# Only the DOWNLOAD is done here (pure CPU, ZeroGPU-safe). The PiD MODEL itself is NOT preloaded — see the
# note below the diffu preload for why that is fatal on ZeroGPU.
prefetch_weights()

# --- The one model this Space ships -----------------------------------------------------------------
# Gabriel/diffu_sd35_128 holds a single `model.safetensors` (128px SD3.5 MMDiT). checkpoints.py accepts a
# .safetensors file and auto-detects qk_norm/unifont from the weights; the ONE non-default arch flag is
# line_height=128 (config.json confirms glyph_line=True, glyph_concat=False, style_in_context=False).
CKPT_REPO = os.environ.get("DIFFU_CKPT_REPO", "Gabriel/diffu_sd35_128")
CKPT_FILE = os.environ.get("DIFFU_CKPT_FILE", "model.safetensors")
ckpt_path = hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILE)

arch = ArchConfig(line_height=128)
device = resolve_device(os.environ.get("DIFFU_STUDIO_DEVICE"))  # autodetects 'zerogpu' via SPACES_ZERO_GPU

# recognizer_name=None: StudioEngine loads the TrOCR read-back with device_map='cuda' in __init__, which
# would init CUDA in the main process and violate ZeroGPU's fork model. The read-back/CER panel is optional.
engine = StudioEngine(device=device, ckpt_root="checkpoints", arch=arch, recognizer_name=None)

# Preload the model in the PARENT process so ZeroGPU's short-lived per-request FORKS reuse one already-loaded
# copy. Without this the model loads lazily INSIDE the first fork, dies with it, and the whole 5.5 GB
# checkpoint (+ VAE + style encoder) reloads on every Generate — the "waiting on model load every time". On
# ZeroGPU `spaces` defers the CUDA transfer, so this is CPU-side at startup; each fork makes it live on the H200.
try:
    engine.model(ckpt_path)
    print("[space] model preloaded — forks reuse it (no per-request reload)", flush=True)
except Exception as exc:  # boot-safe: a preload failure just reverts to lazy per-request loading
    print(f"[space] model preload failed ({type(exc).__name__}: {exc}); lazy-loading per request", flush=True)

# NOTE: PiD is deliberately NOT preloaded here. Unlike the diffu engine (whose CUDA transfer `spaces` defers),
# PiD's loader does REAL CUDA init in the parent process — `instantiate(config.model).cuda()`,
# `model.on_train_start()`, the Gemma-2 encoder's `.to("cuda")`, and `torch.cuda.empty_cache()`. Any real CUDA
# init in the ZeroGPU MAIN process poisons the fork model: every worker fork then dies with
# "RuntimeError: No CUDA GPUs are available" (torch.init in spaces/zero/wrappers.py), breaking BOTH Generate
# and Upscale. So PiD loads lazily INSIDE the first `@spaces.GPU` upscale fork (where CUDA is legal) and is
# cached per fork — the weights are already local (prefetch_weights above), so that first click pays only the
# ~13 s build + ~20 s decode, well inside the 90 s GPU budget.

_here = os.path.dirname(os.path.abspath(__file__))
gallery = sorted(glob.glob(os.path.join(_here, "assets", "styles", "*.png")))

print(f"[space] device={device.label}  ckpt={ckpt_path}  styles={len(gallery)}", flush=True)

# Same launch path as __main__: queued Blocks (NOT gr.Server) so the UI event queue actually runs. On a
# Space, launch() takes no host/port — the platform injects them. NO theme override — the Space must look
# EXACTLY like the local studio (which launches with gradio's default theme).
demo = build_blocks(engine, gallery_paths=gallery or None, default_ckpt=ckpt_path)
demo.queue().launch()
