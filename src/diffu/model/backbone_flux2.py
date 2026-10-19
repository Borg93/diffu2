"""FLUX.2-style backbone = diffusers ``Flux2Transformer2DModel`` (built small, trained from scratch).

The block design of FLUX.2 / Klein: a few dual-stream blocks (separate image / text weights) then mostly
single-stream blocks whose attention and MLP run in parallel off ONE fused input projection, with one
modulation shared by all blocks. Fewer, wider matrix multiplies than the all-dual-stream SD3 MMDiT
(model/backbone.py) — the shape tensor cores want. Same call interface as :class:`Backbone`:
  - latent ``[B, C, h, w]`` -> 2x2 patches -> image tokens (row-major, as SD3's PatchEmbed)
  - context tokens ``[B, L, ctx]`` + the pooled vector appended as one extra (position-free) token —
    Flux2 has no pooled AdaLN input
  - 4-axis RoPE ids ``(t, row, col, l)``: image token = ``(0, row, col, 0)``; the first ``n_content`` text
    tokens = ``(0, 0, j, 0)`` so glyph token j shares the column frequency of image column j (the same
    alignment the SD3 path gets from its shared-column RoPE)
Output: predicted velocity ``[B, C_out, h, w]``.

Reference tokens (``style_mode="canvas"``): the clean reference latent is prepended as its own image tokens,
modulated at t=0 and attending only to itself (diffusers' Flux2 KV processors). Training runs that way
("extract"), so at sampling the reference K/V computed on the first step are reused for every later step
unchanged (:meth:`forward_cached`) — exact, not an approximation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import (
    Flux2KVAttnProcessor,
    Flux2KVCache,
    Flux2KVParallelSelfAttnProcessor,
)

from ..config import BackboneConfig
from .backbone import configure_compile


class _AutocastRMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` whose weight follows the input dtype.

    Under bf16 autocast the Q/K activations arrive in bf16 while the (fp32 master) weight stays fp32, and
    ``rms_norm`` then cannot dispatch to its fused kernel. Casting the weight per call keeps the fused path;
    the parameter itself (and every checkpoint key) is unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(x.dtype) if self.weight is not None else None
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


def _rope_axes(head_dim: int) -> tuple[int, int, int, int]:
    """(t, row, col, l) RoPE channel split: most channels on row/col — t and l are always 0 here."""
    side = (head_dim - 16) // 2
    if head_dim < 32 or side % 2:
        raise ValueError(f"head_dim {head_dim} must be >= 32 with (head_dim - 16) / 2 even")
    return (8, side, side, 8)


class Flux2Backbone(nn.Module):
    """Flux2Transformer2DModel wrapper mapping (latent, timestep, content tokens, pooled) -> velocity."""

    def __init__(
        self,
        cfg: BackboneConfig,
        in_channels: int,
        context_dim: int,
        pooled_dim: int,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        if pooled_dim != context_dim:
            raise ValueError(
                "the flux2 backbone feeds the pooled vector as a context token: pooled_dim == context_dim"
            )
        out_channels = out_channels if out_channels is not None else in_channels
        self.patch = cfg.patch
        self.out_channels = out_channels
        head_dim = cfg.dim // cfg.heads
        self.transformer = Flux2Transformer2DModel(
            patch_size=1,  # we patchify ourselves (2x2 latent patches -> token channels)
            in_channels=in_channels * cfg.patch**2,
            out_channels=out_channels * cfg.patch**2,
            num_layers=cfg.num_layers,
            num_single_layers=cfg.num_single_layers,
            attention_head_dim=head_dim,
            num_attention_heads=cfg.heads,
            joint_attention_dim=context_dim,
            mlp_ratio=cfg.mlp_ratio,
            axes_dims_rope=_rope_axes(head_dim),
            rope_theta=int(cfg.rope_theta),
            guidance_embeds=False,
        )
        # The KV-capable processors behave exactly like the defaults without KV arguments, and implement the
        # reference-token path (t=0 modulation, reference self-attention only, K/V cache) when given them.
        for block in self.transformer.transformer_blocks:
            block.attn.set_processor(Flux2KVAttnProcessor())
        for block in self.transformer.single_transformer_blocks:
            block.attn.set_processor(Flux2KVParallelSelfAttnProcessor())
        for module in self.transformer.modules():
            if type(module) is nn.RMSNorm:
                module.__class__ = _AutocastRMSNorm
        self._ids_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    # --- token layout -------------------------------------------------------------------------------
    def _patchify(self, latent: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latent.shape
        p = self.patch
        x = latent.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, (h // p) * (w // p), c * p * p)  # row-major tokens, (c, py, px) channels

    def _unpatchify(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b = tokens.shape[0]
        p = self.patch
        x = tokens.reshape(b, h // p, w // p, self.out_channels, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, self.out_channels, h, w)

    def _image_ids(self, rows: int, cols: int, col_offset: int, device: torch.device) -> torch.Tensor:
        key = ("img", rows, cols, col_offset, device)
        ids = self._ids_cache.get(key)
        if ids is None:
            ids = torch.zeros(rows, cols, 4, device=device)
            ids[..., 1] = torch.arange(rows, device=device).view(rows, 1)
            ids[..., 2] = torch.arange(cols, device=device).view(1, cols) + col_offset
            ids = ids.reshape(rows * cols, 4)
            self._ids_cache[key] = ids
        return ids

    def _text_ids(self, n_tokens: int, n_content: int | None, device: torch.device) -> torch.Tensor:
        key = ("txt", n_tokens, n_content, device)
        ids = self._ids_cache.get(key)
        if ids is None:
            ids = torch.zeros(n_tokens, 4, device=device)
            if n_content:  # glyph token j sits on canvas column j; any further tokens stay position-free
                ids[:n_content, 2] = torch.arange(n_content, device=device)
            self._ids_cache[key] = ids
        return ids

    def _inputs(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        pooled: torch.Tensor,
        n_content: int | None,
        col_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, h, w = latent.shape
        tokens = self._patchify(latent)
        text = torch.cat([context, pooled.unsqueeze(1).to(context.dtype)], dim=1)
        img_ids = self._image_ids(h // self.patch, w // self.patch, col_offset, latent.device)
        txt_ids = self._text_ids(text.shape[1], n_content, latent.device)
        # Diffu passes the SD3 timestep scale (t·1000); Flux2 multiplies by 1000 itself.
        return tokens, timestep / 1000.0, text, img_ids, txt_ids

    # --- forward variants ------------------------------------------------------------------------------
    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        pooled: torch.Tensor,
        n_content: int | None = None,
    ) -> torch.Tensor:
        _, _, h, w = latent.shape
        tokens, t, text, img_ids, txt_ids = self._inputs(
            latent, timestep, context_tokens, pooled, n_content, 0
        )
        out = self.transformer(
            hidden_states=tokens,
            encoder_hidden_states=text,
            timestep=t,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]
        return self._unpatchify(out, h, w)

    def forward_with_ref(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        pooled: torch.Tensor,
        ref_latent: torch.Tensor,
        n_content: int | None = None,
    ) -> tuple[torch.Tensor, Flux2KVCache]:
        """Velocity for ``latent`` with the clean ``ref_latent`` LEFT of it (columns ``[0, w_ref)``), plus the
        reference K/V cache for :meth:`forward_cached`. Used for every training step and the first sampling
        step. Returns only the target's velocity ``[B, C_out, h, w]``."""
        _, _, h, w = latent.shape
        ref_cols = ref_latent.shape[-1] // self.patch
        tokens, t, text, img_ids, txt_ids = self._inputs(
            latent, timestep, context_tokens, pooled, n_content, ref_cols
        )
        ref_tokens = self._patchify(ref_latent.to(tokens.dtype))
        ref_ids = self._image_ids(h // self.patch, ref_cols, 0, latent.device)
        out, cache = self.transformer(
            hidden_states=torch.cat([ref_tokens, tokens], dim=1),
            encoder_hidden_states=text,
            timestep=t,
            img_ids=torch.cat([ref_ids, img_ids], dim=0),
            txt_ids=txt_ids,
            return_dict=False,
            kv_cache_mode="extract",
            num_ref_tokens=ref_tokens.shape[1],
        )
        return self._unpatchify(out, h, w), cache

    def forward_cached(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        pooled: torch.Tensor,
        cache: Flux2KVCache,
        ref_cols: int,
        n_content: int | None = None,
    ) -> torch.Tensor:
        """Like :meth:`forward_with_ref`, reusing the reference K/V from ``cache`` (the reference is not recomputed)."""
        _, _, h, w = latent.shape
        tokens, t, text, img_ids, txt_ids = self._inputs(
            latent, timestep, context_tokens, pooled, n_content, ref_cols
        )
        out = self.transformer(
            hidden_states=tokens,
            encoder_hidden_states=text,
            timestep=t,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
            kv_cache=cache,
            kv_cache_mode="cached",
        )[0]
        return self._unpatchify(out, h, w)

    # --- shared backbone API ---------------------------------------------------------------------------
    def repa_target(
        self, layer: int, n_text: int, n_ref: int
    ) -> tuple[nn.Module, Callable[[Any], torch.Tensor]]:
        """The block whose output REPA aligns, and how to pull the TARGET image tokens out of that output.

        Layers ``< num_layers`` are dual-stream blocks (output ``(text, image)``; image includes any reference
        tokens first); later ones are single-stream blocks (output ``[text, ref, image]``). ``n_text`` counts
        the pooled token too.
        """
        doubles = self.transformer.transformer_blocks
        singles = self.transformer.single_transformer_blocks
        layer = max(0, min(layer, len(doubles) + len(singles) - 1))
        if layer < len(doubles):
            return doubles[layer], lambda out: out[1][:, n_ref:]
        return singles[layer - len(doubles)], lambda out: out[:, n_text + n_ref :]

    def set_attention_backend(self, backend: str) -> None:
        """Route attention through a diffusers kernel backend (see :meth:`Backbone.set_attention_backend`)."""
        self.transformer.set_attention_backend(backend)

    def enable_optimizations(self) -> None:
        """Gradient checkpointing (non-reentrant, so REPA's hook capture keeps its graph)."""
        self.transformer.enable_gradient_checkpointing()

    def compile_blocks(self) -> None:
        """Regionally ``torch.compile`` the repeated dual- and single-stream blocks (``fullgraph``, dynamic width)."""
        configure_compile()
        self.transformer.compile_repeated_blocks(dynamic=True, fullgraph=True)
