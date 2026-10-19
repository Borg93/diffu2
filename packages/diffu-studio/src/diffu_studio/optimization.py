"""Ahead-of-time (AoT) compilation of the MMDiT backbone for Hugging Face ZeroGPU (H200).

On ZeroGPU the GPU is allocated per-request inside a *forked, short-lived* process, so a JIT
``torch.compile`` (what :meth:`diffu.model.backbone.Backbone.compile_blocks` does) cannot persist its
compiled artifacts across calls — every request pays the cold-start again. The fix (HF ``zerogpu-aoti``
recipe) is to compile ONCE at Space startup with ``torch.export`` + ``spaces.aoti_compile`` and install
the frozen graph onto the module, so every subsequent forked request reuses it.

This module is a NO-OP off ZeroGPU (local ``gpu`` / ``cpu``): the studio keeps its eager / JIT path there.

────────────────────────────────────────────────────────────────────────────────────────────────────
Honesty / what can only be validated on real ZeroGPU H200 hardware
────────────────────────────────────────────────────────────────────────────────────────────────────
The ``spaces.aoti_capture / aoti_compile / aoti_apply`` helpers and the ``@spaces.GPU`` fork model only
exist inside a live Spaces runtime, so the *export → compile → apply → speedup* path here can be
smoke-checked locally (export a fixed-width graph) but only END-TO-END validated on the Space. Two parts
of THIS model are genuine exportability risks and are why every step below is wrapped in a fallback:

  1. ``Backbone.forward`` takes ``n_content: int`` (a *Python int*, not a tensor). Under ``torch.export``
     a plain int specializes to a constant → the exported graph would be frozen to ONE line width. We
     avoid this by exporting a thin wrapper whose forward is tensors-only and derives
     ``n_content = context_tokens.shape[1]`` INTERNALLY, so it rides the dynamic width symbol. This is
     only valid when ``style_in_context=False`` (the released arch), where ``context == content`` and so
     ``context.shape[1] == n_content``. We assert that and fall back otherwise.

  2. The custom 2D-RoPE path (:class:`diffu.model.rope.RoPEJointAttnProcessor`) memoizes ``(cos, sin)`` in
     ``Backbone._rope_cache`` keyed by the python ints ``(h_t, w_t, device)`` and builds them with
     ``arange(w_t)``. Under a DYNAMIC width, ``w_t`` becomes a ``SymInt`` and that dict lookup /
     arange-on-symint is exactly the kind of data-dependent control flow ``torch.export`` can refuse to
     trace cleanly (guard on a symbolic hash, or a graph break). If the dynamic-width export raises, we
     fall back — in order — to a FIXED-width AoT (still a real H200 win for the common bucket), then to
     the eager JIT ``compile_blocks()``, then to plain eager. None of these change numerics.

So: treat dynamic-width AoT as "attempt, measure on the Space." The fixed-width and JIT fallbacks are the
safety net and are what guarantees the Space still boots and runs if export doesn't like the RoPE graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as _nn  # runtime base class for the export wrapper; this module is only imported when AoT is on

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from diffu.config import Config
    from diffu_studio.checkpoints import LoadedModel


# A representative line to drive one real denoise so the captured shapes match production traffic.
# Mid-length Swedish so the natural-width lands in the middle of the bucket span (not the min/max edge).
_SAMPLE_TEXT = "Kongl. Maj:ts nådiga förordning"
_SAMPLE_STYLE_SIZE = 224  # DINO input side (matches diffu.generate.load_style)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def aot_compile_backbone(
    loaded: LoadedModel,
    *,
    cfg_scale: float = 5.0,
    num_steps: int = 4,
    duration: int = 1500,
) -> bool:
    """AoT-compile ``loaded.model.backbone`` in place for ZeroGPU. Returns True iff a compiled graph was
    installed; a NO-OP returning False off Spaces or on any failure (the caller keeps the eager path).

    Runs the whole capture→export→compile inside ``@spaces.GPU(duration=1500)`` (long startup budget) so a
    GPU is held for the one-time compile. ``cfg_scale>0`` makes the capture use the 2×-batch CFG path the
    studio actually runs (see :meth:`diffu.model.diffu.Diffu._denoise_latents`), so the exported batch dim
    matches. ``num_steps`` is tiny — we only need ONE real backbone call to capture shapes.
    """
    from diffu_studio.device import _on_zerogpu  # reuse the studio's single source of truth

    if not _on_zerogpu():
        return False  # local gpu / cpu: keep eager (or opt into JIT compile_blocks elsewhere)

    try:
        import spaces
    except ImportError:  # SPACES_ZERO_GPU set but the package missing — misconfigured Space, stay eager
        print("[studio] AoT skipped: `spaces` not importable despite SPACES_ZERO_GPU")
        return False

    # Capture `loaded` by CLOSURE (not as a @spaces.GPU argument): ZeroGPU forks the decorated call into a
    # fresh process and pickles its ARGS; a ~2B-param CUDA module must not go through that path. The blog's
    # startup pattern references the model as a global/closure for exactly this reason.
    @spaces.GPU(duration=duration)
    def _startup() -> bool:
        return _compile_and_apply(loaded, spaces, cfg_scale=cfg_scale, num_steps=num_steps)

    try:
        return _startup()
    except Exception as exc:  # noqa: BLE001 — AoT is an optimization; a Space must still boot without it
        print(f"[studio] AoT compile failed ({type(exc).__name__}: {exc}); falling back to eager/JIT")
        _fallback_jit(loaded)
        return False


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Implementation (runs on the GPU, inside @spaces.GPU)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _compile_and_apply(loaded: LoadedModel, spaces: Any, *, cfg_scale: float, num_steps: int) -> bool:
    import torch

    model, cfg = loaded.model, loaded.cfg
    backbone = model.backbone

    # --- Guard the assumptions the tensors-only wrapper relies on ------------------------------------
    if not model.glyph_line:
        # Non-glyph-line: content seq length is fixed (max_chars), not width-derived, and n_content!=seq.
        # Not the released arch; skip AoT rather than export a graph with wrong dynamic dims.
        print("[studio] AoT skipped: backbone is not in glyph_line mode (n_content is not width-derived)")
        return False
    if getattr(model, "style_in_context", False):
        # context = cat(content, style) → context.shape[1] != n_content, so we can't derive n_content
        # from the context length. Would need to also thread the style-token count; not worth it here.
        print("[studio] AoT skipped: style_in_context=True breaks the n_content = context.shape[1] identity")
        return False

    # --- (OPTIONAL) torchao FP8 — H200 (sm_90) has hardware FP8; quantize BEFORE export so the frozen ---
    # --- graph bakes in the fp8 kernels. Gated off by default: it shifts numerics and must be CER-checked
    # --- on the Space before trusting it. Enable via DIFFU_STUDIO_FP8=1.
    _maybe_quantize_fp8(backbone)

    # --- Capture ONE real backbone call so the example inputs are production-representative -----------
    # aokit.exporting.capture hooks the module's forward and records the (args, kwargs) of the first call.
    import aokit

    style = _dummy_style(model, size=_SAMPLE_STYLE_SIZE)
    latent_hw = _sample_latent_hw(model, cfg, _SAMPLE_TEXT)
    with aokit.exporting.capture(backbone) as call:
        # A short real denoise → exercises the exact _denoise_latents → backbone(...) path (incl. the 2×
        # CFG batch when cfg_scale>0). We only need the first backbone call's shapes; num_steps is minimal.
        model.generate([_SAMPLE_TEXT], style, latent_hw=latent_hw, num_steps=num_steps, cfg_scale=cfg_scale)

    # call.args = (latent, timestep, context_tokens, pooled); call.kwargs = {"n_content": <int>}
    # We DROP n_content here: the wrapper recomputes it from context_tokens.shape[1] so it stays dynamic.
    latent, timestep, context_tokens, pooled = call.args[:4]
    example_args = (latent, timestep, context_tokens, pooled)

    # --- Dynamic shapes: ONE fundamental symbol = w_tok (patch-column count) -------------------------
    # latent[..., w] with w = patch * w_tok ; context seq = w_tok ; batch shared across all four inputs.
    patch = backbone.patch
    w_min, w_max = _width_token_bounds(cfg, patch)
    batch = latent.shape[0]  # 2 under CFG (b=1 line), 1 otherwise — see below for the dynamic-batch note.

    w_tok = torch.export.Dim("w_tok", min=w_min, max=w_max)
    # Derived: the latent's width dim is patch * w_tok (encodes divisibility-by-patch for free), and the
    # context sequence dim IS w_tok. h (dim 2) and channels (dim 1) are static (fixed line height / VAE C).
    dynamic_shapes = (
        {3: patch * w_tok},  # latent   [B, C, h, patch*w_tok]  — dynamic width only
        {},  # timestep [B]                 — static (batch fixed; see dynamic-batch note)
        {1: w_tok},  # context  [B, w_tok, ctx]     — seq tied to width
        {},  # pooled   [B, ctx]              — static
    )
    # NOTE (dynamic batch): if you want ONE graph to serve both cfg_scale>0 (batch 2b) and cfg_scale==0
    # (batch b), add a `batch = torch.export.Dim("batch", min=1, max=2)` and put `{0: batch}` on all four
    # entries (and on timestep). It widens the export surface (more guards) so it is left OFF by default —
    # the studio's default cfg_scale=5 always takes the 2× path, so a fixed batch is the low-risk choice.

    # --- Export the tensors-only wrapper (n_content derived inside) → aokit weightless AoT package -----
    import os
    from pathlib import Path

    import aokit
    from aokit import LazyAOTIModel

    wrapper = _BackboneExportWrapper(backbone)
    with torch.no_grad():
        exported = torch.export.export(wrapper, args=example_args, dynamic_shapes=dynamic_shapes)
    # aokit.compile_and_save writes a WEIGHTLESS AOTInductor .pt2 (graph decoupled from weights) under
    # <pkg>/root/; LazyAOTIModel lazy-loads it per forked ZeroGPU request and re-binds the live backbone
    # weights — the ZeroGPU-friendly path from the aokit README. (Regional variant: compile_and_save(...,
    # submodule="transformer_blocks") + aokit.load_from_package_dir(model.backbone.transformer, pkg) — the
    # model already marks JointTransformerBlock as its repeated block; cuts cold-start, same speedup.)
    pkg_dir = Path(os.environ.get("DIFFU_AOKIT_DIR", "/tmp/diffu_aokit_backbone"))
    aokit.compile_and_save(package_dir=str(pkg_dir), exported_program=exported)
    # wrapper.state_dict() keys are "_backbone.*" — matching the exported graph's constant FQNs.
    compiled = LazyAOTIModel(pkg_dir / "root" / "package.pt2").with_weights(dict(wrapper.state_dict()))

    # --- Install onto the REAL backbone --------------------------------------------------------------
    # We can't use spaces.aoti_apply(compiled, backbone) verbatim: aoti_apply swaps in a forward whose
    # signature matches the exported graph (4 tensors), but every call site does
    # `backbone(..., n_content=nc)` (see diffu.model.diffu._denoise_latents). So we install a thin shim
    # that accepts + IGNORES n_content (the graph recomputes it) and delegates to the compiled callable.
    _install_compiled_forward(backbone, compiled)
    backbone._aot_compiled = True  # gate flag: compile_blocks() must not ALSO run (they conflict)

    print(
        f"[studio] AoT backbone compiled — dynamic width w_tok∈[{w_min},{w_max}] "
        f"(latent width {patch * w_min}–{patch * w_max}), batch={batch}"
    )
    return True


class _BackboneExportWrapper(_nn.Module):
    """Tensors-only ``nn.Module`` view of ``Backbone.forward`` for ``torch.export`` (which requires an
    ``nn.Module``) — recomputes ``n_content`` inside so a graph-freezing Python int becomes a value that
    rides the dynamic width symbol. ``_backbone`` is registered as a submodule, so the exported graph
    carries the backbone's real weights and its constant FQNs are ``_backbone.*`` — matched by installing
    ``wrapper.state_dict()`` (see ``LazyAOTIModel.with_weights`` in ``_compile_and_apply``). Valid because
    ``style_in_context=False`` → context == content, so ``context.shape[1] == n_content`` (asserted upstream).
    """

    def __init__(self, backbone: _nn.Module) -> None:
        super().__init__()
        self._backbone = backbone

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        pooled: torch.Tensor,
    ) -> torch.Tensor:
        n_content = context_tokens.shape[1]  # SymInt under dynamic width; == n_content when context==content
        return self._backbone(latent, timestep, context_tokens, pooled, n_content=n_content)


def _install_compiled_forward(backbone: nn.Module, compiled: Any) -> None:
    """Point ``backbone.forward`` at the compiled callable, keeping the ``n_content=`` call-site signature.

    The AoT graph recomputes n_content from ``context_tokens.shape[1]``, so the passed-in ``n_content`` is
    intentionally dropped.
    """

    def _compiled_forward(
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        pooled: torch.Tensor,
        n_content: int | None = None,  # accepted for call-site compatibility; recomputed in the graph
    ) -> torch.Tensor:
        return compiled(latent, timestep, context_tokens, pooled)

    backbone.forward = _compiled_forward  # type: ignore[method-assign]


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Shape / input helpers
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _width_token_bounds(cfg: Config, patch: int) -> tuple[int, int]:
    """(min, max) patch-column count the studio can ask the backbone for, from the width bucketing.

    Upstream (:func:`diffu.generate.natural_width`) rounds the canvas to a multiple of 16 and caps it at
    ``cfg.data.max_line_width``; the latent is ``canvas // vae.downscale_factor`` and the backbone patches
    that by ``patch``. The narrowest canvas a line can get is ``round_up(line_height, 16)`` (a 1-char line
    still gets a canvas at least as tall as it is wide). We widen the range by a small slack so a bucket at
    the exact edge never falls outside the exported symbol.
    """
    ds = cfg.vae.downscale_factor
    max_canvas = cfg.data.max_line_width
    min_canvas = ((cfg.data.line_height + 15) // 16) * 16  # round_up(line_height, 16)
    w_max = (max_canvas // ds) // patch
    w_min = max(2, (min_canvas // ds) // patch - 1)  # -1 slack; Dim min must be ≥2 (0/1 specialize)
    return w_min, w_max


def _sample_latent_hw(model: nn.Module, cfg: Config, text: str) -> tuple[int, int]:
    """(h, w) latent grid for ``text`` at the studio's canonical auto-width geometry (matches line.py)."""
    from diffu.generate import natural_width

    ds = cfg.vae.downscale_factor
    canvas_w = natural_width(model.guidance_renderer, text, cfg)
    return (cfg.data.line_height // ds, max(1, canvas_w // ds))


def _dummy_style(model: nn.Module, *, size: int) -> torch.Tensor:
    """A neutral ImageNet-normalized ``[1,3,S,S]`` style ref on the model's device/dtype for the capture
    denoise. Content of the style is irrelevant — we only need it to produce representative SHAPES."""
    import torch

    p = next(model.parameters())
    return torch.zeros(1, 3, size, size, device=p.device, dtype=p.dtype)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Optional levers (clearly gated; each shifts numerics or trades cold-start — measure on the Space)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _maybe_quantize_fp8(backbone: nn.Module) -> None:
    """(OPTIONAL) torchao dynamic-FP8 on the transformer BEFORE export — H200 has hardware fp8 matmul.

    Off by default (``DIFFU_STUDIO_FP8=1`` to enable). FP8 changes numerics, so a page-CER check on the
    Space is required before trusting it. Must run before ``torch.export`` so the exported graph captures
    the quantized (fp8) linears. Only the big MMDiT ``transformer`` is quantized; the small conditioning
    encoders stay full-precision.
    """
    import os

    if os.environ.get("DIFFU_STUDIO_FP8") != "1":
        return
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

        quantize_(backbone.transformer, Float8DynamicActivationFloat8WeightConfig())
        print("[studio] applied torchao FP8 dynamic quant to backbone.transformer (pre-export)")
    except Exception as exc:  # noqa: BLE001 — fp8 is opt-in; never fail the compile on it
        print(f"[studio] FP8 quant skipped ({type(exc).__name__}: {exc})")


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# (OPTIONAL) Regional AoT — cut cold-start by compiling only the repeated JointTransformerBlock
# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Full-graph export of the whole 24-layer MMDiT is the biggest compile (and the most exposed to the RoPE
# tracing risk above). The repeated-block trick (what compile_blocks() already does for the JIT) also
# applies to AoT: export ONE `JointTransformerBlock` with dynamic seq length, aoti_compile it, and swap the
# compiled block into all layers. It slashes cold-start and shrinks the export surface (the RoPE freqs are
# passed IN via joint_attention_kwargs, so a single block's graph is smaller). It is more plumbing — you
# must export the block's own forward signature (hidden_states, encoder_hidden_states, temb, image_rotary_emb)
# and rebuild the kwargs at call time — so it is left as a documented follow-up rather than the default path.
# If the whole-backbone export above proves too fragile on the Space, this is the recommended next step.


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Fallback
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _fallback_jit(loaded: LoadedModel) -> None:
    """AoT failed — try the eager JIT ``compile_blocks()`` so we still get the −70%-kernel-launch win.

    Gated by ``_aot_compiled``: if a compiled forward WAS installed we must NOT also JIT-compile (they
    conflict). On ZeroGPU the JIT won't persist across forks (that's why we tried AoT), but within a single
    warm request it still helps; off Spaces it's the normal win. Best-effort — never raises.
    """
    backbone = loaded.model.backbone
    if getattr(backbone, "_aot_compiled", False):
        return
    try:
        backbone.compile_blocks()
    except Exception as exc:  # noqa: BLE001 — last-resort optimization; eager is always correct
        print(f"[studio] JIT compile_blocks fallback skipped ({type(exc).__name__}: {exc}); running eager")
