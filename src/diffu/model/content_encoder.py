"""Unifont glyph-line content encoder, native PyTorch.

Renders the target text as one strip of GNU-Unifont glyphs, so ANY Unicode glyph (incl. å ä ö Å Ä Ö and
historical letterforms) is representable by construction. A ResNet18 trunk encodes the strip to a feature
map whose columns become the column-aligned content tokens of the denoiser.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import resnet18

from ..config import ConditioningConfig


def default_unifont() -> str | None:
    """Locate the GNU-Unifont (full BMP coverage) for the glyph stencil: ``$DIFFU_UNIFONT_PATH``, else the font
    shipped in the package (``diffu/assets/unifont.otf``). PIL's default font renders many BMP glyphs — incl.
    Swedish ``Ä Å Ö ä å ö`` — as ``.notdef`` boxes, so there is no fallback to it."""
    env = os.environ.get("DIFFU_UNIFONT_PATH")
    if env and Path(env).is_file():
        return env
    packaged = Path(__file__).resolve().parent.parent / "assets" / "unifont.otf"
    return str(packaged) if packaged.is_file() else None


class UnifontRenderer:
    """Render a character to a ``[1, S, S]`` tensor in ``[0, 1]`` using a GNU-Unifont TTF.

    GNU Unifont covers the full BMP, so every Swedish/historical glyph has a bitmap. Render at
    ``S >= 32`` (not the native 16) so diacritic rings/dots survive the CNN stem.
    """

    def __init__(self, font_path: str | None = None, size: int = 32) -> None:
        self.size = size
        font_path = font_path or default_unifont()
        try:
            if font_path is None:
                raise OSError("no unifont found")
            self.font = ImageFont.truetype(font_path, size)
        except OSError as exc:
            # The model trains on the Unifont stencil; any other font is out-of-distribution and degrades the
            # output to gibberish — fail fast, never render wrong.
            raise RuntimeError(
                f"UnifontRenderer needs GNU Unifont ({exc}). The font ships inside the diffu package "
                f"(diffu/assets/unifont.otf) — a broken/partial install is the likely cause. Override with "
                f"DIFFU_UNIFONT_PATH=/path/to/unifont.otf."
            ) from exc
        # Per-char rasterization is a pure function of (char, font, size); cache it so each glyph is drawn by
        # PIL exactly once instead of on every forward step (the Swedish+historical charset is a few hundred
        # glyphs). Plain dict (not a registered buffer) keyed char -> CPU [1,S,S]; downstream only stacks/cats
        # (which copy), never mutates it, so sharing the cached tensor is safe and byte-identical.
        self._cache: dict[str, torch.Tensor] = {}

    def render_char(self, ch: str) -> torch.Tensor:
        cached = self._cache.get(ch)
        if cached is not None:
            return cached
        img = Image.new("L", (self.size, self.size), color=0)
        ImageDraw.Draw(img).text((0, 0), ch, fill=255, font=self.font)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0
        glyph = arr.unsqueeze(0)  # [1, S, S]
        self._cache[ch] = glyph
        return glyph


class GlyphLineCNN(nn.Module):
    """ResNet18 trunk that keeps the spatial FEATURE MAP (no avgpool/fc) — the line content encoder.

    Input ``[N, 1, H, W]`` (a rendered glyph LINE) -> ``[N, out_dim, H/32, W/32]``, so horizontal layout
    is preserved (column j of the map ≈ horizontal region j of the line). ImageNet-pretrained by default (a
    warm start — ImageNet edge/stroke filters beat random for letter bitmaps); ``pretrained=True`` loads
    ImageNet or raises, never silently random. The 1-channel stem is re-initialized.
    """

    def __init__(self, out_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1-ch glyph (re-init)
        self.body = nn.Sequential(*list(net.children())[1:-2])  # bn,relu,maxpool,layer1..4 -> [N,512,h,w]
        self.proj = nn.Conv2d(512, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(self.stem(x)))  # [N, out_dim, H/32, W/32]


class GlyphLineContentEncoder(nn.Module):
    """text -> ``w_tokens`` COLUMN-aligned content tokens via a whole-line glyph image + CNN feature map.

    Line-level content: render the whole line as one image (glyphs laid out across the
    width), encode to a spatial feature map, then pool the height and resample the WIDTH to exactly
    ``w_tokens`` = the image's patch-column count. Token ``j`` then corresponds to image column ``j`` — the
    alignment the backbone's shared-column RoPE exploits (model/rope.py). No learned positional table is
    needed: position comes from the column resampling + RoPE.
    """

    def __init__(self, cfg: ConditioningConfig, out_dim: int) -> None:
        super().__init__()
        self.max_len = cfg.max_chars
        self.renderer = UnifontRenderer(cfg.unifont_path, size=cfg.glyph_size)
        # channels_last: the ResNet trunk's convs run NHWC (tensor-core-friendly, no per-layer nchwToNhwc
        # layout copies). Numerically identical — a memory layout, not a math change; preserved across .to(device).
        self.cnn = GlyphLineCNN(out_dim, pretrained=cfg.glyph_cnn_pretrained).to(
            memory_format=torch.channels_last
        )

    def _render_line(self, text: str) -> torch.Tensor:
        """Render ``text`` to one ``[1, S, S*len]`` glyph strip (glyphs concatenated left-to-right)."""
        chars = list(text)[: self.max_len] or [" "]
        glyphs = [self.renderer.render_char(c) for c in chars]  # each [1, S, S]
        return torch.cat(glyphs, dim=-1)  # [1, S, S*len]

    def forward(
        self, texts: list[str], device: torch.device, w_tokens: int, extents: list[int] | None = None
    ) -> torch.Tensor:
        """Column-aligned content tokens ``[B, w_tokens, out_dim]``.

        ``extents=None`` stretches each line's glyph features over all ``w_tokens`` columns. With ``extents``
        (``cfg.cond.glyph_pad``), line ``i`` is resampled onto its first ``extents[i]`` columns — where its ink
        actually is — and the remaining columns are blank (zero) tokens.
        """
        lines = [self._render_line(t) for t in texts]  # each [1, S, W_i]
        s = lines[0].shape[-2]
        w = max(line.shape[-1] for line in lines)
        # Train-mode BatchNorm needs more than one value per channel, and one glyph alone reaches the last
        # stage as 1x1: give it a blank second cell, cropped off the feature map again below.
        padded = 2 * s if self.training and len(lines) == 1 and w < 2 * s else w
        batch = torch.zeros(len(texts), 1, s, padded)  # right-pad the batch with blank (0)
        for i, line in enumerate(lines):
            batch[i, :, :, : line.shape[-1]] = line
        dtype = next(self.cnn.parameters()).dtype  # renders are fp32; match the model dtype (bf16 mode)
        # [B, out_dim, h', w']
        fmap = self.cnn(batch.to(device, dtype=dtype, memory_format=torch.channels_last))
        fmap = fmap[..., : max(1, round(fmap.shape[-1] * w / padded))]
        if extents is None:
            # Pool height -> 1 and resample width -> w_tokens, so token j ↔ image patch-column j.
            fmap = F.interpolate(fmap, size=(1, w_tokens), mode="bilinear", align_corners=False)
            return fmap.flatten(2).transpose(1, 2)  # [B, w_tokens, out_dim]
        if len(extents) != len(texts):
            raise ValueError(f"got {len(extents)} extents for {len(texts)} texts")
        out = fmap.new_zeros(len(texts), fmap.shape[1], 1, w_tokens)
        fw = fmap.shape[-1]
        for i, (line, extent) in enumerate(zip(lines, extents, strict=True)):
            valid = max(1, round(fw * line.shape[-1] / w))  # this line's own feature columns (drop batch pad)
            cols = min(max(int(extent), 1), w_tokens)
            out[i, :, :, :cols] = F.interpolate(
                fmap[i : i + 1, :, :, :valid], size=(1, cols), mode="bilinear", align_corners=False
            )[0]
        return out.flatten(2).transpose(1, 2)  # [B, w_tokens, out_dim]
