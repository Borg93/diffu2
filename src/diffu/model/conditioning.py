"""Conditioning for Diffu (native transformers).

- GlyphLineRenderer: text -> estimated line width in pixels (the fill-ratio / auto-width estimator).
- StyleEncoder: DINOv3 ViT-L (frozen) features from 1-5 reference line crops.
- PerceiverResampler: compress style patch tokens -> fixed style tokens + pooled vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from ..config import ConditioningConfig


class GlyphLineRenderer:
    """Width estimator: how many pixels ``text`` should span at this line height.

    Historical name — this class once rendered glyph strips for the removed DOG-guidance path; only
    the width formula survived, and it lives on as ``Diffu.guidance_renderer`` (kept: generate.py,
    diffu-page, diffu-runner and diffu-studio all reach it through that attribute). Its one job now
    is backing auto-width and the fill-ratio scalar via :meth:`natural_width`.
    """

    def __init__(self, cfg: ConditioningConfig, height: int = 64) -> None:
        self.height = height
        self.max_width = cfg.max_chars * (height // 2)

    def natural_width(self, text: str) -> int:
        """Estimated pixel width of ``text`` at this strip height (length-scaled, capped at max_width).

        The single source of truth for "how wide should this line be" — used at inference
        (generate.py / app.py auto-width) and for the fill-ratio scalar (model/diffu.py), so the
        train and inference width signals come from the same formula.
        """
        size = int(self.height * 0.7)
        return min(self.max_width, max(self.height, int(len(text) * size * 0.6)))


class PerceiverResampler(nn.Module):
    """Compress a variable number of patch tokens to a fixed set of latents (Flamingo/IP-Adapter style)."""

    def __init__(self, in_dim: int, dim: int, num_latents: int = 64, heads: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.in_proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "n_lat": nn.LayerNorm(dim),
                        "n_ctx": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                        "n_mlp": nn.LayerNorm(dim),
                        "mlp": nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                    }
                )
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = self.in_proj(x)
        lat = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        for b in self.blocks:
            a, _ = b["attn"](b["n_lat"](lat), b["n_ctx"](ctx), b["n_ctx"](ctx))
            lat = lat + a
            lat = lat + b["mlp"](b["n_mlp"](lat))
        return self.norm(lat)  # [B, num_latents, dim]


class AttentionPool(nn.Module):
    """Pool a token set to one vector via a learned query (vs a plain mean).

    A single learnable query cross-attends over the tokens, so the pooled style vector is a *content-
    aware* weighted summary — it can emphasise the tokens that carry the discriminative style cues
    (slant, stroke width) instead of averaging everything (which dilutes them). Used for the AdaLN
    pooled style vector fed to SD3.
    """

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(tokens.shape[0], -1, -1)  # [B, 1, dim]
        out, _ = self.attn(q, tokens, tokens)  # [B, 1, dim]
        return self.norm(out.squeeze(1))  # [B, dim]


class StyleEncoder(nn.Module):
    """DINOv3 ViT-L (frozen) + Perceiver resampler + attention-pool."""

    def __init__(self, cfg: ConditioningConfig, dim: int) -> None:
        super().__init__()
        try:
            self.proc = AutoImageProcessor.from_pretrained(cfg.style_encoder)
            self.enc = AutoModel.from_pretrained(cfg.style_encoder)
        except Exception as exc:  # noqa: BLE001 - HF raises varied (non-OSError) types for gated repos
            raise RuntimeError(
                f"could not load the style encoder {cfg.style_encoder!r}. DINOv3 is gated: accept its licence "
                "on the Hub and run `hf auth login`."
            ) from exc
        self.enc.requires_grad_(False)
        self.resampler = PerceiverResampler(self.enc.config.hidden_size, dim, cfg.style_tokens)
        self.pool = AttentionPool(dim)  # learned-query pool for the AdaLN vector (trained; see train.py)

    def train(self, mode: bool = True) -> StyleEncoder:
        # Keep the frozen DINO encoder in eval() regardless of the parent's train()/eval() toggling, so its
        # features (style tokens AND the REPA target) stay deterministic — no dropout / stochastic-depth even
        # if a future checkpoint ships nonzero rates. requires_grad_(False) alone does not change train mode.
        super().train(mode)
        self.enc.eval()
        return self

    @torch.no_grad()
    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.enc(pixel_values=pixel_values).last_hidden_state  # [B, P, D]

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self._features(pixel_values)
        tokens = self.resampler(feats)  # [B, K, dim]
        pooled = self.pool(tokens)  # [B, dim] — learned-query attention pool (vs the old mean)
        return tokens, pooled
