"""Backbone = diffusers SD3 MMDiT (SD3Transformer2DModel) with 2D RoPE.

We deliberately do NOT hand-roll the transformer: diffusers' implementation is optimized (attention
backends, gradient checkpointing, torch.compile-friendly) and battle-tested. We only wrap it to map our
conditioning:
  - hidden_states         = noised line latent (+ conditioning channels)  [B, C_in, h, w]
  - encoder_hidden_states = column-aligned glyph tokens                    [B, L, context_dim]
  - pooled_projections    = pooled AdaLN vector                            [B, context_dim]
  - timestep              = flow-matching timestep                         [B]
Output: predicted velocity [B, C, h, w].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers import SD3Transformer2DModel
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PatchEmbed

from ..config import BackboneConfig
from .rope import RoPEJointAttnProcessor, build_2d_rope

_RECOMPILE_LIMIT = 256


def configure_compile(mode: str | None = None) -> None:
    """Dynamo / Inductor settings for the regional, dynamic-width compiles (blocks and VAE).

    torch keeps these overrides per thread, so they hold only in the thread that sets them:
    :func:`ensure_compile_config` re-applies them wherever a compiled region runs on another thread.
    """
    import torch._dynamo
    import torch._inductor.config as inductor_config
    import torch.fx.experimental._config as fx_config

    # use_duck_shape=False keeps the line width a free symbol, so a width that coincidentally equals
    # batch/heads/h_t doesn't trigger a recompile.
    fx_config.use_duck_shape = False
    # Widths don't recompile (dynamic=True), but each grad mode (train / no_grad / inference) compiles its own
    # graphs for the first block (its context is a view), the middle blocks and the last (context_pre_only),
    # and batch size 1 specializes: past the default recompile_limit of 8, which fullgraph=True makes fatal.
    torch._dynamo.config.recompile_limit = _RECOMPILE_LIMIT
    torch._dynamo.config.accumulated_recompile_limit = 2048
    if mode is not None and mode.startswith("max-autotune"):
        # The Inductor settings diffusers' SD3 docs pair with max-autotune ("Performance Optimizations for SD3").
        inductor_config.conv_1x1_as_mm = True
        inductor_config.coordinate_descent_tuning = True
        inductor_config.epilogue_fusion = False
        inductor_config.coordinate_descent_check_all_directions = True


def ensure_compile_config(mode: str | None = None) -> None:
    """Apply :func:`configure_compile` in the CURRENT thread unless it already holds there: a server's worker
    threads, and autograd's backward thread, where gradient checkpointing re-runs a compiled block."""
    import torch._dynamo

    if torch._dynamo.config.recompile_limit != _RECOMPILE_LIMIT:
        configure_compile(mode)


class Backbone(nn.Module):
    """SD3 Transformer2D wrapper mapping (latent, timestep, context tokens, pooled) -> velocity.

    The SD3 absolute ``pos_embed`` table is replaced by a position-free ``PatchEmbed`` and a custom 2D-RoPE
    joint-attention processor, so variable-width lines have no width cap; per-call rotary freqs are passed
    via ``joint_attention_kwargs`` (grad-checkpoint-safe — SD3 threads that dict through every block and
    through its gradient-checkpointing wrapper).
    """

    def __init__(self, cfg: BackboneConfig, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        # in_channels may exceed out_channels when conditioning channels are concatenated onto the input
        # (the canvas reference): the model READS extra channels but still predicts only the velocity.
        out_channels = out_channels if out_channels is not None else in_channels
        self.transformer = SD3Transformer2DModel(
            patch_size=cfg.patch,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=cfg.num_layers,
            attention_head_dim=cfg.dim // cfg.heads,
            num_attention_heads=cfg.heads,
            joint_attention_dim=cfg.context_dim,
            caption_projection_dim=cfg.dim,
            pooled_projection_dim=cfg.context_dim,
            pos_embed_max_size=1,  # its sincos table is replaced right below; size 1 skips building ~150 MB
            qk_norm=cfg.qk_norm,  # SD3.5: RMSNorm on Q/K (training stability)
            dual_attention_layers=cfg.dual_attention_layers,  # SD3.5-medium / MMDiT-X (() = off)
        )
        self.patch = cfg.patch
        self.head_dim = cfg.dim // cfg.heads
        self.theta = cfg.rope_theta
        # A position-free PatchEmbed: no absolute sincos table, so no width cap (RoPE carries position).
        self.transformer.pos_embed = PatchEmbed(
            patch_size=cfg.patch, in_channels=in_channels, embed_dim=cfg.dim, pos_embed_type=None
        )
        self.transformer.set_attn_processor(RoPEJointAttnProcessor())
        # Memoized 2D-RoPE freqs keyed by (h_tokens, w_tokens, device) — see forward().
        self._rope_cache: dict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._compiled = False  # compile_blocks() ran: forward re-applies the per-thread compile settings
        self._compile_mode: str | None = None

    def _rope(self, h_t: int, w_t: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Memoized 2D-RoPE (cos, sin) for an ``h_t x w_t`` grid (constants, so caching is bit-identical).

        Kept in fp32 whatever the model dtype: ``apply_rotary_emb`` computes in fp32 anyway, and fp32 tables
        make bf16 inference rotate exactly like (fp32-weight) training instead of with bf16-rounded angles.
        """
        key = (h_t, w_t, device)
        cached = self._rope_cache.get(key)
        if cached is None:
            # Never an inference tensor, even when a grid is first seen while sampling (generate() runs under
            # inference_mode): autograd can't save one for backward, so the next training step on it would fail.
            with torch.inference_mode(False):
                cached = build_2d_rope(h_t, w_t, self.head_dim, device, self.theta)
            self._rope_cache[key] = cached
        return cached

    def enable_gradient_checkpointing(self) -> None:
        """diffusers' gradient checkpointing, non-reentrant (so REPA's hook capture still backprops).

        The recompute runs on autograd's backward thread, so a compiled block first re-applies the compile
        settings there (they are per thread; a block would otherwise recompile against the default limit).
        """

        def checkpoint(block: nn.Module, *args: object) -> object:
            def run(*inputs: object) -> object:
                if self._compiled:
                    ensure_compile_config(self._compile_mode)
                return block(*inputs)

            return torch.utils.checkpoint.checkpoint(run, *args, use_reentrant=False)

        self.transformer.enable_gradient_checkpointing(checkpoint)

    def set_attention_backend(self, backend: str) -> None:
        """Route attention through a diffusers kernel backend (``RoPEJointAttnProcessor`` dispatches).

        E.g. ``"_native_cudnn"``, ``"flash"`` / ``"flash_hub"`` (FlashAttention, local install / Hub kernel),
        ``"sage_blackwell_hub"``. Benchmark on the target GPU: support varies by architecture, and the Sage
        kernels are inference-only (no backward). Call before :meth:`compile_blocks`.
        """
        self.transformer.set_attention_backend(backend)

    def fuse_qkv_projections(self) -> None:
        """Inference: one Q/K/V matmul per stream instead of three (diffusers' ``Attention.fuse_projections``).

        Keeps :class:`RoPEJointAttnProcessor` (``SD3Transformer2DModel.fuse_qkv_projections`` would swap in
        the stock processor and drop our RoPE). Call AFTER loading weights and NOT with a LoRA attached —
        the fused copies are taken from the base Q/K/V weights; the unfused ones are kept for training.
        """
        for module in self.transformer.modules():
            if isinstance(module, Attention):
                module.fuse_projections()

    def compile_blocks(self, mode: str | None = None) -> None:
        """Regionally ``torch.compile`` the repeated ``JointTransformerBlock`` (shared by train + inference).

        All 24 blocks share one compiled code object (3 graphs per grad mode: the first, middle and last block
        differ), so cold start is cheap and per-step throughput matches a full compile. The change is IN-PLACE (no ``_orig_mod.`` state-dict prefix, no ``OptimizedModule`` proxy) so a REPA
        forward-hook on an inner block still fires with grad intact. ``dynamic=True`` keeps the variable line
        width on one symbolic shape — pair with width bucketing (train) / one width per call (inference) so
        only a handful of shapes occur and recompiles stay bounded. Measured: −70% kernel launches, ~1.9×
        train throughput (docs/PERF_AUDIT.md §0e).

        ``mode`` is a ``torch.compile`` mode: ``None`` (default), ``"max-autotune-no-cudagraphs"`` (GEMM
        autotuning; training-safe), ``"reduce-overhead"`` / ``"max-autotune"`` (add CUDA graphs — inference
        only; one graph per line width, so keep widths bucketed).
        """
        configure_compile(mode)
        self._compiled, self._compile_mode = True, mode
        # Compiled, the per-call Q/K/V weight concat fuses into the bf16 weight cast autocast does anyway, so
        # the one-GEMM projection is free; eager it would add a weight copy per call, so it stays off there.
        for processor in self.transformer.attn_processors.values():
            if isinstance(processor, RoPEJointAttnProcessor):
                processor.concat_qkv = True
        # SD3 ships `_repeated_blocks` empty, so populate it or compile_repeated_blocks() is a no-op.
        self.transformer._repeated_blocks = ["JointTransformerBlock"]
        # fullgraph=True so a graph break in the custom RoPE processor (apply_rotary_emb on a slice + cat,
        # qk RMSNorm, SDPA) fails LOUDLY at compile time instead of silently un-fusing and erasing the win.
        self.transformer.compile_repeated_blocks(dynamic=True, fullgraph=True, mode=mode)

    def forward(
        self, latent: torch.Tensor, timestep: torch.Tensor, context_tokens: torch.Tensor, pooled: torch.Tensor
    ) -> torch.Tensor:
        if self._compiled:  # the compile settings are per thread: apply them in this one too
            ensure_compile_config(self._compile_mode)
        # The 2D-RoPE (cos, sin) are deterministic constants of the token grid (arange-derived,
        # non-differentiable), so memoize them per (grid, device): under width-bucketing only a handful of
        # grids occur, so this rebuilds arange/get_1d_rotary/cat at most once per bucket instead of every step.
        h_t = latent.shape[-2] // self.patch
        w_t = latent.shape[-1] // self.patch
        # The context is the column-aligned glyph tokens: one row of columns, rotated with the SAME column
        # frequencies as the image patches (shared-column RoPE, MSRoPE-style), so glyph token j lines up with
        # image column j.
        joint_attention_kwargs = {
            "image_rotary_emb": self._rope(h_t, w_t, latent.device),
            "content_rotary_emb": self._rope(1, context_tokens.shape[1], latent.device),
        }
        return self.transformer(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=context_tokens,
            pooled_projections=pooled,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
