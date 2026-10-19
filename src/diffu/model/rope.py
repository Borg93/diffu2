"""2D Rotary Position Embedding (RoPE) for the SD3 image tokens.

Diffu's lines are variable-width, so SD3's absolute (sincos) ``pos_embed`` table — which is
capped at ``pos_embed_max_size`` patches and must extrapolate off-table for wide lines — is a poor
positional prior (the root cause of the short-text-on-wide-canvas "overflow"). We drop that table
(see backbone.py) and instead rotate the IMAGE-token Q/K with a relative 2D RoPE.

Two pieces live here:
  * :func:`build_2d_rope` — builds the (cos, sin) frequency tensors for an ``h x w`` patch grid. The
    grid is flattened ROW-MAJOR (h-then-w) to match ``PatchEmbed.forward``'s ``flatten(2).transpose(1, 2)``
    (``[B, C, h, w] -> [B, h*w, C]``), so token ``i`` carries the freqs of patch ``(i // w, i % w)``.
    Each axis (row index, col index) gets ``head_dim // 2`` rotary channels; the two axes are
    concatenated to a full ``head_dim``. This is exactly how diffusers' ``FluxPosEmbed`` composes a
    multi-axis rotary, so we reuse its public helpers verbatim.
  * :class:`RoPEJointAttnProcessor` — a faithful copy of diffusers 0.41.0 ``JointAttnProcessor2_0``
    that additionally rotates the image Q/K with 2D-RoPE. By default the TEXT (content + style) tokens
    stay position-free. With the line-glyph path (``content_rotary_emb`` + ``n_content`` threaded in),
    the FIRST ``n_content`` text tokens (the column-aligned glyph-line tokens) are rotated with a
    shared-column RoPE built for ``(1 x n_content)`` — so glyph token ``j`` carries the SAME column
    frequency as image patches in column ``j`` (Qwen-Image MSRoPE-style text<->image alignment). The
    remaining (style) tokens stay position-free.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed


def build_2d_rope(
    h_tokens: int,
    w_tokens: int,
    head_dim: int,
    device: torch.device,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 2D RoPE (cos, sin) for an ``h x w`` patch grid flattened row-major.

    Args:
        h_tokens: Number of patch rows.
        w_tokens: Number of patch columns.
        head_dim: Per-head channel dim; must be even (each axis gets ``head_dim // 2`` channels).
        device: Device for the position indices (and thus the returned freqs).
        theta: RoPE base frequency.

    Returns:
        ``(cos, sin)`` each ``[h_tokens * w_tokens, head_dim]`` (float32). The first ``head_dim // 2``
        channels encode the ROW (h) index, the rest encode the COLUMN (w) index.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even (each axis uses head_dim // 2), got {head_dim}")
    # Row-major flatten to match PatchEmbed's flatten(2).transpose(1, 2): token i -> patch (i//w, i%w).
    hh = torch.arange(h_tokens, device=device).view(h_tokens, 1).expand(h_tokens, w_tokens).reshape(-1)
    ww = torch.arange(w_tokens, device=device).view(1, w_tokens).expand(h_tokens, w_tokens).reshape(-1)
    half = head_dim // 2
    cos_h, sin_h = get_1d_rotary_pos_embed(
        half, hh.float(), theta=theta, use_real=True, repeat_interleave_real=True
    )
    cos_w, sin_w = get_1d_rotary_pos_embed(
        half, ww.float(), theta=theta, use_real=True, repeat_interleave_real=True
    )
    cos = torch.cat([cos_h, cos_w], dim=-1)  # [h*w, head_dim]
    sin = torch.cat([sin_h, sin_w], dim=-1)  # [h*w, head_dim]
    return cos, sin


class RoPEJointAttnProcessor:
    """SD3 joint-attention processor that rotates the IMAGE Q/K with 2D RoPE.

    A faithful copy of diffusers 0.41.0 ``JointAttnProcessor2_0.__call__`` semantics (our config has
    ``norm_q``/``norm_k``/``norm_added_*`` = None and no ``attn2`` path) with one addition: if an
    ``image_rotary_emb`` is supplied (threaded in via ``joint_attention_kwargs``), it is applied to the
    image-token query and key — BEFORE the text query/key are concatenated, so only image tokens are
    rotated and the text tokens stay position-free.

    Unlike the stock SD3 processor, the attention itself goes through diffusers' ``dispatch_attention_fn``,
    so ``transformer.set_attention_backend(...)`` (cuDNN / FlashAttention / Sage kernels) applies to it —
    ``set_attention_backend`` only touches processors exposing these two attributes. ``None`` = the native
    ``scaled_dot_product_attention`` default, identical to the previous direct call.
    """

    _attention_backend = None
    _parallel_config = None
    # One GEMM per stream by concatenating the Q/K/V weights per call (set by Backbone.compile_blocks): the
    # training-time counterpart of Backbone.fuse_qkv_projections, with the same parameters / checkpoints.
    concat_qkv = False

    def _qkv(self, x: torch.Tensor, q: nn.Module, k: nn.Module, v: nn.Module) -> tuple[torch.Tensor, ...]:
        """Q, K, V of ``x``: one matmul over the concatenated weights when they are plain Linears (a LoRA-
        wrapped projection is not, so it keeps three); gradients flow back into each original weight."""
        layers = (q, k, v)
        if self.concat_qkv and all(type(layer) is nn.Linear for layer in layers):
            biases = [layer.bias for layer in layers]
            if all(b is None for b in biases) or all(b is not None for b in biases):
                weight = torch.cat([layer.weight for layer in layers])
                bias = None if biases[0] is None else torch.cat(biases)
                return F.linear(x, weight, bias).chunk(3, dim=-1)
        return q(x), k(x), v(x)

    def __call__(
        self,
        attn: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        content_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        n_content: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # image (`sample`) projections — one fused matmul after Backbone.fuse_qkv_projections()
        fused = getattr(attn, "fused_projections", False)
        if fused:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = self._qkv(hidden_states, attn.to_q, attn.to_k, attn.to_v)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # diffusers' RMSNorm returns fp32 when its weight is fp32 (training: fp32 master weights under bf16
        # autocast), which would carry fp32 Q/K through RoPE, the concat and into attention. Cast back to the
        # activation dtype (V's): a no-op in bf16 inference, and training then matches inference numerics.
        if attn.norm_q is not None:
            query = attn.norm_q(query).to(value.dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(value.dtype)

        # 2D RoPE on the IMAGE tokens only (before the text tokens are concatenated below).
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=2)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=2)

        # text (`context`) projections — position-free (no rotary)
        if encoder_hidden_states is not None:
            if fused:
                enc_q, enc_k, enc_v = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
            else:
                enc_q, enc_k, enc_v = self._qkv(
                    encoder_hidden_states, attn.add_q_proj, attn.add_k_proj, attn.add_v_proj
                )

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q).to(enc_v.dtype)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k).to(enc_v.dtype)

            # Shared-column RoPE on the first n_content (glyph-line) text tokens only; style stays
            # position-free. Split/rotate/concat (not in-place) to keep autograd intact.
            if content_rotary_emb is not None and n_content:
                cq, sq = enc_q[:, :, :n_content], enc_q[:, :, n_content:]
                ck, sk = enc_k[:, :, :n_content], enc_k[:, :, n_content:]
                cq = apply_rotary_emb(cq, content_rotary_emb, sequence_dim=2)
                ck = apply_rotary_emb(ck, content_rotary_emb, sequence_dim=2)
                enc_q = torch.cat([cq, sq], dim=2)
                enc_k = torch.cat([ck, sk], dim=2)

            query = torch.cat([query, enc_q], dim=2)  # [image, text] along seq dim
            key = torch.cat([key, enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

        # dispatch_attention_fn takes [B, S, H, D]; the transposes are views, not copies.
        hidden_states = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(value.dtype)  # the activation dtype (query.dtype could be fp32)

        if encoder_hidden_states is not None:
            # image tokens are the first residual.shape[1]; the rest are text
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)  # linear proj
        hidden_states = attn.to_out[1](hidden_states)  # dropout

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states
