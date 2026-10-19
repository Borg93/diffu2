"""Backbone = diffusers SD3 MMDiT (SD3Transformer2DModel).

We deliberately do NOT hand-roll the transformer: diffusers' implementation is more optimized
(fused / SDPA = FlashAttention, gradient checkpointing, torch.compile-friendly) and battle-tested.
We only wrap it to map our conditioning:
  - hidden_states         = noised line latent           [B, C, h, w]
  - encoder_hidden_states = content + style tokens        [B, L+K, context_dim]   (joint attention)
  - pooled_projections    = pooled style vector           [B, pooled_dim]          (AdaLN)
  - timestep              = flow-matching timestep        [B]
Output: predicted velocity [B, C, h, w].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from diffusers import SD3Transformer2DModel
from diffusers.models.embeddings import PatchEmbed

from ..config import BackboneConfig
from .rope import RoPEJointAttnProcessor, build_2d_rope


def configure_compile() -> None:
    """Process-global Dynamo settings for the regional, dynamic-width block compile (both backbones)."""
    import torch._dynamo
    import torch.fx.experimental._config as fx_config

    # use_duck_shape=False keeps the line width a free symbol, so a width that coincidentally equals
    # batch/heads/h_t doesn't trigger a recompile. Process-global; set once per entrypoint.
    fx_config.use_duck_shape = False
    # Variable widths + the last block's context_pre_only=True guard trigger several recompiles; the
    # default recompile_limit (8) is too low and Dynamo ABORTS. Raise it (bucketing keeps the real count
    # to ~num_buckets; dynamic=True keeps most widths on one symbolic shape).
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 2048


class Backbone(nn.Module):
    """SD3 Transformer2D wrapper mapping (latent, timestep, content+style tokens) -> velocity.

    With ``cfg.rope`` (default) the SD3 absolute ``pos_embed`` table is replaced by a position-free
    ``PatchEmbed`` and a custom 2D-RoPE joint-attention processor; per-call rotary freqs are passed via
    ``joint_attention_kwargs`` (grad-checkpoint-safe — SD3 threads that dict through every block and
    through its gradient-checkpointing wrapper). With ``cfg.rope=False`` the stock SD3 pos_embed is kept.
    """

    def __init__(
        self,
        cfg: BackboneConfig,
        in_channels: int,
        context_dim: int,
        pooled_dim: int,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        # in_channels may exceed out_channels when a glyph latent is channel-concatenated onto the input
        # (glyph_concat): the model READS extra channels but still predicts only the latent_channels velocity.
        out_channels = out_channels if out_channels is not None else in_channels
        self.transformer = SD3Transformer2DModel(
            sample_size=cfg.sample_size,
            patch_size=cfg.patch,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=cfg.num_layers,
            attention_head_dim=cfg.dim // cfg.heads,
            num_attention_heads=cfg.heads,
            joint_attention_dim=context_dim,
            caption_projection_dim=cfg.dim,
            pooled_projection_dim=pooled_dim,
            pos_embed_max_size=cfg.pos_embed_max_size,
            qk_norm=cfg.qk_norm,  # SD3.5: RMSNorm on Q/K (training stability)
            dual_attention_layers=cfg.dual_attention_layers,  # SD3.5-medium / MMDiT-X (() = off)
        )
        self.rope = cfg.rope
        self.patch = cfg.patch
        self.head_dim = cfg.dim // cfg.heads
        self.theta = cfg.rope_theta
        if self.rope:
            # Swap in a position-free PatchEmbed: drops the absolute sincos table AND the
            # pos_embed_max_size width cap, so variable-width lines patch-embed without a hard limit.
            self.transformer.pos_embed = PatchEmbed(
                height=cfg.sample_size,
                width=cfg.sample_size,
                patch_size=cfg.patch,
                in_channels=in_channels,
                embed_dim=cfg.dim,
                pos_embed_type=None,
            )
            self.transformer.set_attn_processor(RoPEJointAttnProcessor())
            # Memoized 2D-RoPE freqs keyed by (h_tokens, w_tokens, device, dtype) — see forward().
            self._rope_cache: dict[
                tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
            ] = {}

    def _rope(
        self, h_t: int, w_t: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Memoized 2D-RoPE (cos, sin) for an ``h_t x w_t`` grid (constants, so caching is bit-identical)."""
        key = (h_t, w_t, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is None:
            cos, sin = build_2d_rope(h_t, w_t, self.head_dim, device, self.theta)
            cached = (cos.to(dtype), sin.to(dtype))
            self._rope_cache[key] = cached
        return cached

    def enable_optimizations(self) -> None:
        """Enable diffusers' gradient checkpointing (non-reentrant).

        Non-reentrant checkpointing keeps the checkpointed blocks' outputs attached to the autograd
        graph during the forward pass, so REPA's forward-hook capture of a mid-block hidden state can
        still backprop into the backbone (reentrant checkpointing detaches it -> REPA would train only
        its projection head). diffusers (0.38+) defaults to non-reentrant for torch>=1.11, so the plain
        call already gives us what REPA needs. (The previous ``gradient_checkpointing_kwargs=`` form
        was never a diffusers arg — it is a HF Transformers arg — so that try-branch always raised.)
        """
        self.transformer.enable_gradient_checkpointing()

    def repa_target(
        self, layer: int, n_text: int, n_ref: int
    ) -> tuple[nn.Module, Callable[[Any], torch.Tensor]]:
        """The block whose output REPA aligns, and how to pull the image tokens out of it (``(text, image)``).

        ``n_text`` / ``n_ref`` exist for the flux2 backbone's interface; SD3 keeps a canvas reference in the
        image grid itself, which the caller slices by column.
        """
        blocks = self.transformer.transformer_blocks
        block = blocks[max(0, min(layer, len(blocks) - 1))]
        return block, lambda out: out[1] if isinstance(out, tuple | list) and len(out) > 1 else out

    def set_attention_backend(self, backend: str) -> None:
        """Route attention through a diffusers kernel backend (``RoPEJointAttnProcessor`` dispatches).

        E.g. ``"_native_cudnn"``, ``"flash"`` / ``"flash_hub"`` (FlashAttention, local install / Hub kernel),
        ``"sage_blackwell_hub"``. Benchmark on the target GPU: support varies by architecture, and the Sage
        kernels are inference-only (no backward). Call before :meth:`compile_blocks`.
        """
        self.transformer.set_attention_backend(backend)

    def compile_blocks(self) -> None:
        """Regionally ``torch.compile`` the repeated ``JointTransformerBlock`` (shared by train + inference).

        Compiles ONE block (not all 24): cold start is cheap and per-step throughput matches a full compile.
        The change is IN-PLACE (no ``_orig_mod.`` state-dict prefix, no ``OptimizedModule`` proxy) so a REPA
        forward-hook on an inner block still fires with grad intact. ``dynamic=True`` keeps the variable line
        width on one symbolic shape — pair with width bucketing (train) / one width per call (inference) so
        only a handful of shapes occur and recompiles stay bounded. Measured: −70% kernel launches, ~1.9×
        train throughput (docs/PERF_AUDIT.md §0e).
        """
        configure_compile()
        # SD3 ships `_repeated_blocks` empty, so populate it or compile_repeated_blocks() is a no-op.
        self.transformer._repeated_blocks = ["JointTransformerBlock"]
        # fullgraph=True so a graph break in the custom RoPE processor (apply_rotary_emb on a slice + cat,
        # qk RMSNorm, SDPA) fails LOUDLY at compile time instead of silently un-fusing and erasing the win.
        self.transformer.compile_repeated_blocks(dynamic=True, fullgraph=True)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        pooled: torch.Tensor,
        n_content: int | None = None,
    ) -> torch.Tensor:
        joint_attention_kwargs: dict[str, object] | None = None
        if self.rope:
            # The 2D-RoPE (cos, sin) are deterministic constants of the token grid (arange-derived,
            # non-differentiable), so memoize them per (grid, device, dtype): under width-bucketing only a
            # handful of grids occur, so this rebuilds arange/get_1d_rotary/cat at most once per bucket
            # instead of every step. Bit-identical to rebuilding. (apply_rotary_emb upcasts internally; the
            # dtype cast is just a clean device/dtype match, cached alongside.)
            h_t = latent.shape[-2] // self.patch
            w_t = latent.shape[-1] // self.patch
            joint_attention_kwargs = {"image_rotary_emb": self._rope(h_t, w_t, latent.device, latent.dtype)}
            if n_content:
                # Shared-column RoPE for the line-glyph content tokens: 1 row × n_content columns, so
                # content token j carries the SAME column frequency as image patches in column j (MSRoPE-style).
                joint_attention_kwargs["content_rotary_emb"] = self._rope(
                    1, n_content, latent.device, latent.dtype
                )
                joint_attention_kwargs["n_content"] = n_content
        return self.transformer(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=context_tokens,
            pooled_projections=pooled,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
