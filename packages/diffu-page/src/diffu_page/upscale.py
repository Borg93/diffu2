"""PiD 4× super-resolution — turn a composed page into a crisp high-res scan. THE shared implementation.

This is the single source of truth for PiD upscaling: the interactive studio (``diffu_studio.upscale``
re-exports these) and the Ray page pipeline (``diffu_runner.pages.PidActor``) both call in here, so the
tricky PiD-load logic lives in exactly one place and can never drift between them.

Runs NVIDIA PiD's *from-clean* path (input image → the model's own Flux VAE-encode → PiD 4-step pixel
decode). No Z-Image / LDM pipeline is involved: PiD carries its own VAE and text encoder, so the upscaler
never touches the text2image stack. The degrade ``sigma`` is ``0.0`` (clean input, no hallucination).

Import-safe everywhere: nothing here imports torch, ``pid`` or CUDA at module load — a CPU box, a run
without the PiD checkpoints, or a Space that hasn't installed PiD must still be able to import this module.
If PiD is unavailable (or the upscale raises), :func:`pid_upscale` LOGS and returns the input image
unchanged, so a caller can always try it. Model construction hardcodes ``.cuda()`` inside PiD, so the whole
thing must run where CUDA is available — inside a ``@spaces.GPU`` fork on ZeroGPU, or a Ray GPU actor.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

_WEIGHTS_REPO = "nvidia/PiD"  # HF Hub weights (the code comes from the pinned nv-tlabs/PiD git dependency)
# PINNED revision — the repo's later revisions DELETED the flux 2kto4k distill checkpoint below (superseded
# by v1pt5 variants our pinned pid code doesn't know), so an unpinned download resolves a snapshot without
# our weights and load_pid FileNotFounds. Bump this together with the pid git pin, never alone.
_WEIGHTS_REVISION = "b4887b3c8fc65277a9b7a084292bf9fc0778c5ac"
_CKPT_TYPE = "2kto4k"  # trained 1024→3840 (dynamic_shift) — the right band for a whole large page
# Whole PAGE → feed the 2kto4k checkpoint; both checkpoints are 4×, this one is tuned for long side ≤ 1024.
_CKPT_PATTERNS = [
    "checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/*",
    "checkpoints/ae.safetensors",
]

# Gemma-2-2b-it is PiD's frozen text encoder; transformers pulls it from this repo at model build time.
_TEXT_ENCODER_REPO = "Efficient-Large-Model/gemma-2-2b-it"
_TEXT_ENCODER_PATTERNS = ["*.safetensors*", "*.json", "tokenizer*", "*.model"]

# Built lazily on the FIRST GPU call and cached for the process (multi-GB: Gemma-2-2b + Flux VAE + PiD net).
_STATE: dict[str, Any] = {}


def pid_available() -> bool:
    """True if the ``pid`` package is importable (a pinned git dependency). No torch/CUDA touched."""
    return importlib.util.find_spec("pid") is not None


def prefetch_weights() -> None:
    """Download the PiD weights (decoder + VAE + Gemma-2 text encoder) into the HF cache. CPU-safe, no torch.

    Call this at startup on ZeroGPU: the first Upscale click runs inside a time-budgeted ``@spaces.GPU``
    window, which must not be spent on an ~8 GB download. A no-op when everything is already cached; logged
    and skipped (the caller falls back gracefully) when offline without a cache. Does NOT touch CUDA, so it
    is safe to call in a ZeroGPU main process (unlike :func:`load_pid`, which must run inside the GPU fork).
    """
    if not pid_available():
        return
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(_WEIGHTS_REPO, revision=_WEIGHTS_REVISION, allow_patterns=_CKPT_PATTERNS)
        snapshot_download(_TEXT_ENCODER_REPO, allow_patterns=_TEXT_ENCODER_PATTERNS)
        logger.info("PiD weights prefetched into the HF cache.")
    except Exception:  # noqa: BLE001 — prefetch is an optimization; the lazy load path remains authoritative
        logger.exception("PiD weight prefetch failed — the first upscale will try to download instead.")


def _apply_transformers_compat() -> None:
    """Re-apply the PiD demo's legacy transformers<5 shims ONLY if the pre-5.0 symbols are actually present.

    The demo (pinned to transformers 4.57) monkeypatched ``masking_utils._vmap_for_bhqkv`` and Gemma-2's
    ``forward`` to dodge a ``torch.vmap`` mask bug. On our transformers 5.x that symbol is gone and
    ``sdpa_mask`` already defaults to the non-vmap path, so the fix is unnecessary — and blindly assigning the
    old attribute would either be dead or crash. Everything is guarded, so on 5.x this function is a no-op.
    """
    try:
        import transformers.masking_utils as mu

        if hasattr(mu, "_vmap_for_bhqkv"):  # 4.5x only; paste the demo's replacement here if you must run 4.x
            logger.debug("pid compat: transformers<5 masking API present; leaving upstream default in place")
    except Exception:  # noqa: BLE001 — a probe; never fail the load on a compat check
        logger.debug("pid compat: masking_utils probe skipped", exc_info=True)
    try:  # importing the module is enough to confirm Gemma-2 loads cleanly; no forward patch on 5.x
        from transformers.models.gemma2 import modeling_gemma2 as _g2  # noqa: F401
    except Exception:  # noqa: BLE001
        logger.debug("pid compat: gemma2 probe skipped", exc_info=True)


def load_pid(*, compile_model: bool | None = None) -> Any:
    """Lazily build + cache the PiD 2kto4k model (VAE + text encoder included). MUST run where CUDA is live.

    The model carries the frozen Flux VAE (``model.vae_encoder``) and the Gemma-2 text encoder, so no Z-Image
    pipeline is loaded. PiD's own ``load_model_from_checkpoint`` resolves ``config_file`` by turning a
    cwd-relative *file path* into a module string (``pid/_src/.../config.py`` → ``pid._src...config``), which
    only works from a repo checkout — ``pid`` is an installed package here, so this imports the config module
    directly and replays the loader's steps. Deliberately NOT replayed: the loader's global
    ``set_random_seed`` / cudnn-benchmark / TF32 flags — process-wide state a co-resident line generator
    depends on; PiD seeds its sampler with an explicit per-call ``torch.Generator`` anyway.

    ZeroGPU note: this does REAL CUDA init (``.cuda()``, ``on_train_start``, the Gemma ``.to("cuda")``,
    ``empty_cache``). It must therefore run INSIDE the ``@spaces.GPU`` fork, never in the main process —
    doing it in the parent poisons the fork model ("No CUDA GPUs are available" in every worker).

    ``compile_model`` arms PiD's OWN built-in ``torch.compile`` support (``PixelDiTModel.enable_compile``
    — the same mechanism its official CLI's ``--compile`` flag uses): compilation is lazy and cached per
    output ``(H, W)`` (``dynamic=False``, one compiled net per exact shape), since
    ``precompute_positional_caches`` must run per-shape before compiling. Our callers all upscale at a
    fixed ``max_side`` from same-size composed pages, so in practice this is one compile, not a recompile
    storm. Incompatible with context-parallel / the encoder-decoder net path (asserted inside
    ``enable_compile``) — neither applies to the standard 2kto4k SR checkpoint we load here.

    ``compile_model=None`` (the default) resolves to ``False`` on Spaces (``SPACE_ID`` set) and ``True``
    elsewhere. Two reasons Spaces must stay EAGER: (a) JIT compile is the wrong tool on ZeroGPU — the
    ``@spaces.GPU`` fork is short-lived, so the compile cost is paid per request (HF's own guidance:
    use ``spaces.aoti_*`` ahead-of-time compilation instead — a possible future upgrade for this path);
    (b) the Space's torch version hard-fails tracing PiD's ``fetch_pos(..., x.device)`` (Dynamo
    ``ConstantVariable`` assertion on ``torch.device``), observed live 2026-07-13. Validated ON for the
    local torch: 10.9s → 4.9s per 4× page pass. :func:`pid_upscale` additionally falls back to a fresh
    EAGER model if a compiled pass raises, so a broken dynamo can degrade speed but never the output.
    """
    model = _STATE.get("model")
    if model is not None:
        return model

    import torch
    from huggingface_hub import snapshot_download
    from pid._ext.imaginaire.lazy_config import instantiate
    from pid._ext.imaginaire.utils.config_helper import override
    from pid._ext.imaginaire.utils.easy_io import easy_io
    from pid._src.inference.checkpoint_registry import get_pid_checkpoint

    _apply_transformers_compat()
    snap = snapshot_download(_WEIGHTS_REPO, revision=_WEIGHTS_REVISION, allow_patterns=_CKPT_PATTERNS)
    ae_path = os.path.join(snap, "checkpoints", "ae.safetensors")

    ck = get_pid_checkpoint("zimage", _CKPT_TYPE)  # "zimage" aliases the Flux 16-ch VAE checkpoints
    config = importlib.import_module("pid._src.configs.pid.config").make_config()
    # vae_pth defaults to "./checkpoints/ae.safetensors" (cwd-relative) — point it at the HF-cached file.
    # `++`: FluxVAEConfig only sets `name` (FluxVAEInterface swallows vae_pth via **kwargs), so the key is
    # absent from the composed config and a plain override raises "not in struct" — append-or-override it.
    config = override(
        config, ["--", f"experiment={ck.experiment}", f"++model.config.tokenizer.vae_pth={ae_path}"]
    )
    config.validate()
    config.freeze()  # type: ignore  # attached at runtime by PiD's make_freezable decorator
    if hasattr(config.model.config, "fsdp_shard_size"):
        config.model.config.fsdp_shard_size = 1  # single-GPU inference, no FSDP
    model = instantiate(config.model).cuda()
    model.on_train_start()
    # strict=False: the LQ-projection keys are expected-missing in the published EMA checkpoint.
    model.load_state_dict(easy_io.load(os.path.join(snap, ck.checkpoint_path)), strict=False)
    model = model.to(dtype=model.precision)
    model.eval()
    torch.cuda.empty_cache()
    if compile_model is None:
        compile_model = "SPACE_ID" not in os.environ  # Spaces stay eager — see the docstring
    if compile_model and hasattr(model, "enable_compile"):
        model.enable_compile()  # lazy; actually compiles on the first pid_upscale() call per (H, W)

    _STATE["model"] = model
    _STATE["compiled"] = bool(compile_model)
    logger.info("PiD %s loaded (%s, compile=%s)", _CKPT_TYPE, ck.experiment, compile_model)
    return model


def _decode_one(model: Any, x: Any, caption: str, *, scale: int, steps: int, seed: int) -> Any:
    """One full PiD pass: [1,3,H,W] bf16 cuda in [-1,1] → [3,H*scale,W*scale] float32 CPU in [-1,1]."""
    import torch

    lq_latent = model.encode_lq_latent(x)  # [1,16,H/8,W/8]
    comp = int(model.vae_encoder.spatial_compression_factor)  # 8
    target_hw = (lq_latent.shape[-2] * comp * scale, lq_latent.shape[-1] * comp * scale)
    batch = {
        model.config.input_caption_key: [caption],
        "LQ_latent": lq_latent.to(dtype=torch.bfloat16, device="cuda"),
        "degrade_sigma": torch.tensor([0.0], device="cuda", dtype=torch.float32),  # clean input, no noise
    }
    out = model.generate_samples_from_batch(
        batch, cfg_scale=1, num_steps=steps, seed=seed, shift=None, image_size=target_hw
    )  # [1,3,1,H*scale,W*scale] in [-1,1]
    return out[0, :, 0].float().clamp(-1, 1).cpu()


def pid_upscale(
    pil_image: Image | None,
    prompt: str = "",
    *,
    max_side: int = 1024,
    scale: int = 4,
    steps: int = 4,
    seed: int = 0,
) -> Image | None:
    """4× super-resolve a clean page image with PiD. Fallback-safe: returns the input unchanged on any failure.

    ``prompt`` is optional (the distilled path bakes CFG out, so guidance/negative prompts are ignored).
    Without the PiD install this is a logged no-op that hands back ``pil_image``, so the caller never breaks.
    Must be invoked where CUDA is available (a ``@spaces.GPU`` fork on ZeroGPU, or a Ray GPU actor).
    The page works at ``max_side`` in ONE pass, exactly like the official demo (1024 → a 4096-class output) —
    it needs ~24 GiB free (≈9 GiB weights + activations), i.e. a ZeroGPU H200 or a mostly-free local GPU.
    """
    if pil_image is None:
        return pil_image
    if not pid_available():
        logger.warning("PiD not installed (add the `pid` git dependency) — returning input unchanged.")
        return pil_image
    try:
        import numpy as np
        import torch
        from PIL import Image as PILImage

        model = load_pid()
        img = pil_image.convert("RGB")

        # Resize the long side to <= max_side, then crop each side to a multiple of 16 (VAE ×8 · patch ×2).
        w, h = img.size
        r = min(1.0, max_side / max(w, h))
        w, h = max(16, (int(w * r) // 16) * 16), max(16, (int(h * r) // 16) * 16)
        img = img.resize((w, h), PILImage.Resampling.LANCZOS)

        x = (
            torch.from_numpy(np.asarray(img, np.uint8).astype("float32"))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(127.5)
            .sub(1.0)
        )  # [1,3,H,W] in [-1,1]
        x = x.to(device="cuda", dtype=torch.bfloat16)

        try:
            out = _decode_one(model, x, prompt or "", scale=scale, steps=steps, seed=seed)
        except Exception:
            if not _STATE.get("compiled"):
                raise
            # A compiled pass can die on a torch whose dynamo can't trace PiD (seen live: the Space's
            # torch asserts on `x.device` in fetch_pos). Rebuild EAGER once and retry — slow beats broken.
            logger.exception("compiled PiD pass failed — rebuilding eager and retrying once")
            _STATE.pop("model", None)
            model = load_pid(compile_model=False)
            out = _decode_one(model, x, prompt or "", scale=scale, steps=steps, seed=seed)
        arr = out.clamp(-1, 1).add(1).mul(0.5).mul(255).round().byte()  # [3,H*4,W*4] → uint8
        return PILImage.fromarray(arr.permute(1, 2, 0).numpy())
    except Exception:  # noqa: BLE001 — an upscale is a bonus; a failure must never lose the user's page
        logger.exception("PiD upscale failed — returning the input image unchanged.")
        return pil_image
    finally:
        try:  # either way, hand the cached activation blocks back — the GPU is shared with other jobs
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup only; never mask the real outcome
            pass
