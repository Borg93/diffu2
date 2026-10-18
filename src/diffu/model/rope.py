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
  * :class:`RoPEJointAttnProcessor` — a faithful copy of diffusers 0.38.0 ``JointAttnProcessor2_0``
    that additionally rotates the image Q/K with 2D-RoPE. By default the TEXT (content + style) tokens
    stay position-free. With the line-glyph path (``content_rotary_emb`` + ``n_content`` threaded in),
    the FIRST ``n_content`` text tokens (the column-aligned glyph-line tokens) are rotated with a
    shared-column RoPE built for ``(1 x n_content)`` — so glyph token ``j`` carries the SAME column
    frequency as image patches in column ``j`` (Qwen-Image MSRoPE-style text<->image alignment). The
    remaining (style) tokens stay position-free.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
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

    A faithful copy of diffusers 0.38.0 ``JointAttnProcessor2_0.__call__`` semantics (our config has
    ``norm_q``/``norm_k``/``norm_added_*`` = None and no ``attn2`` path) with one addition: if an
    ``image_rotary_emb`` is supplied (threaded in via ``joint_attention_kwargs``), it is applied to the
    image-token query and key — BEFORE the text query/key are concatenated, so only image tokens are
    rotated and the text tokens stay position-free.
    """

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

        # image (`sample`) projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 2D RoPE on the IMAGE tokens only (before the text tokens are concatenated below).
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=2)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=2)

        # text (`context`) projections — position-free (no rotary)
        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

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

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

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
