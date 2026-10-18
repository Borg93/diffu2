"""Width-bucketing collate for variable-width handwriting lines.

Lines share a fixed height but vary in width, so a batch must be padded to a common width before
stacking. We right-pad with white (1.0 in the ``[-1, 1]`` range) up to the batch max width rounded
to a multiple of ``width_multiple`` (keeps the latent grid divisible by the VAE downscale × patch),
capped at ``max_width``. Style references are already a fixed size, so they stack directly.

Optional white-pad augmentation (``white_pad_prob`` > 0, TRAIN SPLIT ONLY): before bucketing, each
line is independently right-padded with extra white with probability ``white_pad_prob``. This puts
short text on a wider white canvas at train time so the model learns to leave the tail blank — the
exact regime the fill-ratio scalar (model/diffu.py) explains. Val/test pass ``white_pad_prob=0`` so
their canvases stay text-tight for honest metrics.
"""

from __future__ import annotations

import random
from typing import Any


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def collate_lines(
    batch: list[dict[str, Any]],
    *,
    width_multiple: int = 16,
    max_width: int = 1024,
    white_pad_prob: float = 0.0,
    white_pad_max_frac: float = 0.5,
    rng: random.Random | None = None,
    white_latent_col: Any | None = None,  # torch.Tensor [C, h, 1]; typed Any to keep this module torch-free at import
    downscale: int = 8,
) -> dict[str, Any]:
    """Collate dataset items into a training batch.

    Args:
        batch: Items from ``HandwritingLineDataset`` (image / text / writer_id / style_pixel_values).
        width_multiple: Padded width is rounded up to this (16 = VAE f8 × patch 2).
        max_width: Hard cap on width; wider lines are cropped.
        white_pad_prob: Per-line probability of right-padding with extra white BEFORE bucketing
            (train-split augmentation; 0 = off). Val/test must pass 0 so canvases stay text-tight.
        white_pad_max_frac: Max extra white width as a fraction of the line's natural width.
        rng: RNG for the augmentation (inject for determinism); defaults to the module ``random``.
        white_latent_col: Encoder white-paper latent column ``[C, h, 1]`` (from ``LatentCache``). When the
            items carry a precomputed ``"latent"`` (``--cache-latents`` path), each line's latent is
            right-padded with this column to the batch latent-width and stacked into ``"latents"``. Padding
            in latent space with the encoder's white column (not zeros) matches encoding a white-padded
            image, so cached-latent training stays faithful to the live-encode path.
        downscale: VAE spatial downscale factor (f8 → 8); maps the image width to the latent width.

    Returns:
        ``{"images": [B,3,H,W], "texts": list[str], "style_pixel_values": [B,3,S,S], "style_texts": list[str]}``
        plus ``"latents": [B,C,h,W//downscale]`` when the items carry cached ``"latent"`` tensors.
    """
    import torch
    from torch.nn import functional as F

    rand = rng or random
    images = [b["image"] for b in batch]
    if white_pad_prob > 0.0:  # train-only: simulate short-text-on-wide-canvas before bucketing
        augmented = []
        for im in images:
            w = im.shape[-1]
            if rand.random() < white_pad_prob:
                extra = rand.randint(0, int(white_pad_max_frac * w))
                if extra > 0:
                    im = F.pad(im, (0, extra), value=1.0)  # right-pad with white
            augmented.append(im)
        images = augmented

    target_w = min(max_width, _round_up(max(im.shape[-1] for im in images), width_multiple))
    padded = []
    for im in images:
        w = im.shape[-1]
        if w >= target_w:
            padded.append(im[..., :target_w])
        else:
            padded.append(F.pad(im, (0, target_w - w), value=1.0))  # right-pad with white
    out: dict[str, Any] = {
        "images": torch.stack(padded),
        "texts": [str(b["text"]) for b in batch],
        "style_pixel_values": torch.stack([b["style_pixel_values"] for b in batch]),
        # Display-only metadata for the live eval (the style ref's OWN text, so the sample sheet can
        # show "style wrote B, we asked A"). .get keeps synthetic-data callers (smoketest) working
        # without the key.
        "style_texts": [str(b.get("style_text", "")) for b in batch],
    }
    if white_latent_col is not None and "latent" in batch[0]:
        # Cached-latent path (--cache-latents): right-pad each line's latent to the batch latent-width
        # (target_w // downscale) with the encoder's white-paper column, then stack. Mirrors the image
        # white-padding above, but in latent space (zeros would not match a white-padded encode).
        target_lw = target_w // downscale
        wcol = white_latent_col.to(batch[0]["latent"].dtype)  # [C, h, 1]
        lat_padded = []
        for b in batch:
            lat = b["latent"][..., :target_lw]  # [C, h, w] (crop if a wide line exceeds the bucket)
            pad = target_lw - lat.shape[-1]
            if pad > 0:
                lat = torch.cat([lat, wcol.expand(-1, -1, pad)], dim=-1)
            lat_padded.append(lat)
        out["latents"] = torch.stack(lat_padded)  # [B, C, h, target_lw]
    return out
