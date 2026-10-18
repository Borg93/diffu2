"""Unifont glyph content encoder, native PyTorch.

Renders each character of the target string to a GNU-Unifont bitmap, so ANY Unicode glyph
(incl. å ä ö Å Ä Ö and historical letterforms) is representable by construction. A small CNN
encodes each glyph; a Transformer mixes them into content tokens for the denoiser.
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
    """Locate a bundled GNU-Unifont (full BMP coverage) for the glyph stencil, else None.

    PIL's default font renders many BMP glyphs — incl. Swedish ``Ä Å Ö ä å ö`` — as ``.notdef`` boxes,
    silently corrupting the content signal. Resolution order: ``$DIFFU_UNIFONT_PATH``, then the font
    SHIPPED IN THE PACKAGE (``diffu/assets/unifont.otf`` — present in wheels, so ``pip install`` users
    get the correct stencil with zero setup), then a repo-level ``assets/unifont.{otf,ttf}``.
    """
    env = os.environ.get("DIFFU_UNIFONT_PATH")
    if env and Path(env).is_file():
        return env
    packaged = Path(__file__).resolve().parent.parent / "assets" / "unifont.otf"
    if packaged.is_file():
        return str(packaged)
    for parent in Path(__file__).resolve().parents:
        for name in ("unifont.otf", "unifont.ttf"):
            candidate = parent / "assets" / name
            if candidate.is_file():
                return str(candidate)
    return None


class UnifontRenderer:
    """Render a character to a ``[1, S, S]`` tensor in ``[0, 1]`` using a GNU-Unifont TTF.

    GNU Unifont covers the full BMP, so every Swedish/historical glyph has a bitmap. Render at
    ``S >= 32`` (not the native 16) so diacritic rings/dots survive the CNN stem.
    """

    def __init__(self, font_path: str | None = None, size: int = 32, use_unifont: bool = True) -> None:
        self.size = size
        # The stencil font MUST match what the model trained on. use_unifont=True -> GNU-Unifont (full
        # BMP); False -> PIL's default (what models trained before Unifont was wired, e.g. exp_base_weekend,
        # learned). Feeding the wrong font is out-of-distribution and the output degrades to gibberish.
        if font_path is None and use_unifont:
            font_path = default_unifont()
        self.font: ImageFont.ImageFont | ImageFont.FreeTypeFont
        loaded: ImageFont.FreeTypeFont | None = None
        if font_path is not None:
            try:
                loaded = ImageFont.truetype(font_path, size)
            except OSError:
                loaded = None
        if loaded is None and use_unifont:
            # The model TRAINED on the Unifont stencil; feeding PIL's default font instead is
            # out-of-distribution and degrades output to gibberish — fail fast, never render wrong.
            # (When use_unifont=False the PIL default is intentional — pre-Unifont checkpoints.)
            why = "no unifont found" if font_path is None else f"could not load {font_path!r}"
            raise RuntimeError(
                f"UnifontRenderer needs GNU Unifont but {why}. The font ships inside the diffu package "
                f"(diffu/assets/unifont.otf) — a broken/partial install is the likely cause. Override "
                f"with DIFFU_UNIFONT_PATH=/path/to/unifont.otf, or pass use_unifont=False ONLY for "
                f"checkpoints trained without Unifont."
            )
        self.font = loaded if loaded is not None else ImageFont.load_default()
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

    def render_string(self, text: str, max_len: int) -> torch.Tensor:
        glyphs = [self.render_char(c) for c in list(text)[:max_len]]
        while len(glyphs) < max_len:
            glyphs.append(torch.zeros(1, self.size, self.size))
        return torch.stack(glyphs, dim=0)  # [max_len, 1, S, S]


class GlyphCNN(nn.Module):
    """Per-glyph ResNet18 stem -> feature vector.

    Content encoder: a 1-channel input conv + ResNet18 trunk. The trunk is
    ImageNet-pretrained by default (a warm start — ImageNet edge/stroke filters beat random for letter
    bitmaps). ``pretrained=False`` = pure random scratch. Binary: ``pretrained=True`` loads ImageNet
    weights and **raises** if they can't be fetched — it never silently degrades to random.
    (The first conv is replaced below with a 1-channel conv, so only the trunk inherits ImageNet.)
    """

    def __init__(self, out_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1-ch glyph (re-init)
        net.fc = nn.Identity()  # ResNet18 trunk -> 512-d feature
        self.backbone = net
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))  # [N, out_dim]


class GlyphContentEncoder(nn.Module):
    """text strings -> content tokens ``[B, L, out_dim]`` via Unifont render + CNN + Transformer."""

    def __init__(self, cfg: ConditioningConfig, out_dim: int) -> None:
        super().__init__()
        self.max_len = cfg.max_chars
        self.renderer = UnifontRenderer(cfg.unifont_path, size=cfg.glyph_size, use_unifont=cfg.use_unifont)
        self.cnn = GlyphCNN(out_dim, pretrained=cfg.glyph_cnn_pretrained)
        # Learnable positional embedding over character slots. WITHOUT this the TransformerEncoder
        # (and SD3, which adds no position to encoder_hidden_states) is permutation-invariant, so the
        # glyph tokens are an unordered set and the model cannot place characters in order -> texture
        # but illegible text. The content tokens must carry position so characters land in order.
        self.pos = nn.Parameter(torch.randn(1, self.max_len, out_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(out_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.out_dim = out_dim

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        # Pad only to the LONGEST text in this batch (capped at max_len), not always to max_len. This
        # keeps content-token compute proportional to the actual text length, so max_chars can be large
        # (e.g. 128/256 for long lines) without every short line carrying a tail of blank-glyph tokens.
        eff_len = min(self.max_len, max((len(t) for t in texts), default=1))
        eff_len = max(eff_len, 1)
        batch = torch.stack([self.renderer.render_string(t, eff_len) for t in texts])  # [B,L,1,S,S]
        b, ln, _, s, _ = batch.shape
        dtype = next(self.cnn.parameters()).dtype  # renders are fp32; match the model dtype (bf16 mode)
        feats = self.cnn(batch.view(b * ln, 1, s, s).to(device, dtype=dtype)).view(b, ln, self.out_dim)
        feats = feats + self.pos[:, :ln]  # inject character order (pos table sized for max_len)
        return self.transformer(feats)  # [B, L, out_dim]


class GlyphLineCNN(nn.Module):
    """ResNet18 trunk that keeps the spatial FEATURE MAP (no avgpool/fc) — the line content encoder.

    Input ``[N, 1, H, W]`` (a rendered glyph LINE) -> ``[N, out_dim, H/32, W/32]``, so horizontal layout
    is preserved (column j of the map ≈ horizontal region j of the line). ImageNet-pretrained by default
    (same binary contract as ``GlyphCNN``: loads ImageNet or raises; never silently random).
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
        self.renderer = UnifontRenderer(cfg.unifont_path, size=cfg.glyph_size, use_unifont=cfg.use_unifont)
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

    def forward(self, texts: list[str], device: torch.device, w_tokens: int) -> torch.Tensor:
        lines = [self._render_line(t) for t in texts]  # each [1, S, W_i]
        s = lines[0].shape[-2]
        w = max(line.shape[-1] for line in lines)
        batch = torch.zeros(len(texts), 1, s, w)  # right-pad the batch with blank (0)
        for i, line in enumerate(lines):
            batch[i, :, :, : line.shape[-1]] = line
        dtype = next(self.cnn.parameters()).dtype  # renders are fp32; match the model dtype (bf16 mode)
        fmap = self.cnn(
            batch.to(device, dtype=dtype, memory_format=torch.channels_last)
        )  # [B, out_dim, h', w']
        # Pool height -> 1 and resample width -> w_tokens, so token j ↔ image patch-column j.
        fmap = F.interpolate(fmap, size=(1, max(w_tokens, 1)), mode="bilinear", align_corners=False)
        return fmap.flatten(2).transpose(1, 2)  # [B, w_tokens, out_dim]


class GlyphLatentEncoder(nn.Module):
    """text -> a spatial glyph LATENT ``[B, Cg, h, w]`` to CHANNEL-CONCAT onto the noisy latent.

    Fuse content into the input: render the whole line, encode it with a small conv
    block to a feature map the size of the diffusion latent, and stack it onto the noisy latent. The
    glyph is then present at every cell (impossible to ignore) and inherits the image's RoPE for free.
    The render is Unicode-safe (Unifont per-char concat, so å ä ö survive). The final ``F.interpolate``
    guarantees the output matches the target latent grid exactly (robust to conv rounding).
    """

    def __init__(self, cfg: ConditioningConfig, out_ch: int) -> None:
        super().__init__()
        self.max_len = cfg.max_chars
        self.renderer = UnifontRenderer(cfg.unifont_path, size=cfg.glyph_size, use_unifont=cfg.use_unifont)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 3, stride=2, padding=1),
        )

    def _render_line(self, text: str) -> torch.Tensor:
        chars = list(text)[: self.max_len] or [" "]
        glyphs = [self.renderer.render_char(c) for c in chars]  # each [1, S, S]
        return torch.cat(glyphs, dim=-1)  # [1, S, S*len]

    def forward(self, texts: list[str], device: torch.device, hw: tuple[int, int]) -> torch.Tensor:
        h, w = hw
        lines = [self._render_line(t) for t in texts]  # each [1, S, W_i]
        s = lines[0].shape[-2]
        wmax = max(line.shape[-1] for line in lines)
        batch = torch.zeros(len(texts), 1, s, wmax)  # right-pad the batch with blank (0)
        for i, line in enumerate(lines):
            batch[i, :, :, : line.shape[-1]] = line
        dtype = next(self.net.parameters()).dtype  # renders are fp32; match the model dtype (bf16 mode)
        fmap = self.net(batch.to(device, dtype=dtype))  # [B, Cg, ~s/8, ~wmax/8]
        return F.interpolate(fmap, size=(h, w), mode="bilinear", align_corners=False)  # [B, Cg, h, w]
