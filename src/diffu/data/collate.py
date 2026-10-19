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

Every batch carries ``widths`` — each line's real pixel width before any padding (its ink extent; the
``glyph_pad`` content signal). Items carrying a ``style_image`` (``style_mode="canvas"``) also yield
``ref_images`` (white-padded to one shared width, so the reference/target boundary is the same latent
column for the whole batch), ``ref_widths`` and ``ref_texts``. A reference wider than ``max_ref_width``
is cropped and its transcription blanked — the text would no longer match the pixels.
"""

from __future__ import annotations

import random
from typing import Any


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def collate_lines(
    batch: list[dict[str, Any]],
    *,
    width_multiple: int = 16,
    max_width: int = 1024,
    white_pad_prob: float = 0.0,
    white_pad_max_frac: float = 0.5,
    max_ref_width: int = 1024,
) -> dict[str, Any]:
    """Collate dataset items into a training batch.

    Args:
        batch: Items from ``HandwritingLineDataset`` (image / text, + style_pixel_values or style_image).
        width_multiple: Padded width is rounded up to this (16 = VAE f8 × patch 2).
        max_width: Hard cap on width; wider lines are cropped.
        white_pad_prob: Per-line probability of right-padding with extra white BEFORE bucketing
            (train-split augmentation; 0 = off). Val/test must pass 0 so canvases stay text-tight.
        white_pad_max_frac: Max extra white width as a fraction of the line's natural width.
        max_ref_width: Cap on the in-canvas reference width (``style_image`` items only).

    Returns:
        ``{"images": [B,3,H,W], "texts": list[str], "widths": list[int], "style_texts": list[str]}`` plus
        ``"style_pixel_values": [B,3,S,S]`` (dino-mode items), and ``ref_images`` / ``ref_widths`` /
        ``ref_texts`` when they carry a ``style_image``.
    """
    import torch
    from torch.nn import functional as F

    images = [b["image"] for b in batch]
    widths = [im.shape[-1] for im in images]  # real extent, BEFORE the white-pad augmentation
    if white_pad_prob > 0.0:  # train-only: simulate short-text-on-wide-canvas before bucketing
        augmented = []
        for im in images:
            w = im.shape[-1]
            if random.random() < white_pad_prob:
                extra = random.randint(0, int(white_pad_max_frac * w))
                if extra > 0:
                    im = F.pad(im, (0, extra), value=1.0)  # right-pad with white
            augmented.append(im)
        images = augmented

    target_w = min(max_width, round_up(max(im.shape[-1] for im in images), width_multiple))
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
        "widths": [min(w, target_w) for w in widths],
        # Display-only metadata for the live eval (the style ref's OWN text, so the sample sheet can
        # show "style wrote B, we asked A"). .get keeps synthetic-data callers (smoketest) working
        # without the key.
        "style_texts": [str(b.get("style_text", "")) for b in batch],
    }
    if "style_pixel_values" in batch[0]:  # the DINO crop (style_mode="dino"; canvas items carry style_image)
        out["style_pixel_values"] = torch.stack([b["style_pixel_values"] for b in batch])
    if "style_image" in batch[0]:
        out.update(_collate_refs(batch, width_multiple, max_ref_width))
    return out


def _collate_refs(batch: list[dict[str, Any]], width_multiple: int, max_ref_width: int) -> dict[str, Any]:
    """Pad the in-canvas references to one shared width (white), cropping any past ``max_ref_width``."""
    import torch
    from torch.nn import functional as F

    refs = [b["style_image"] for b in batch]
    texts = [str(b.get("style_text", "")) for b in batch]
    ref_w = min(
        round_up(max_ref_width, width_multiple), round_up(max(r.shape[-1] for r in refs), width_multiple)
    )
    padded, widths = [], []
    for i, r in enumerate(refs):
        if r.shape[-1] > ref_w:
            r = r[..., :ref_w]
            texts[i] = ""  # cropped: the transcription no longer matches the pixels
        widths.append(r.shape[-1])
        padded.append(F.pad(r, (0, ref_w - r.shape[-1]), value=1.0))
    return {"ref_images": torch.stack(padded), "ref_widths": widths, "ref_texts": texts}
