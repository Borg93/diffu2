"""Diagnostic: does lowering CFG (or tightening width) cure the iam_lora character-looping?

Generates a few IAM texts through the iam_lora adapter at several cfg_scales, plus a tight-width
variant, and composes a labelled grid so we can SEE whether the looping is a sampling artifact
(fixable at inference) or a training-quality failure (needs a retrain with --log-cer).
"""

from __future__ import annotations

import os

import torch
from PIL import Image, ImageDraw, ImageFont

from diffu.config import Config
from diffu.generate import _round_up, load_checkpoint, load_style, to_pil
from diffu.lora import load_lora
from diffu.model import Diffu

OUT = "/tmp/claude-1000/-home-blackwell-Desktop-diffu/6245d0dc-db0d-4483-891b-dddc762c7c70/scratchpad/cfg_sweep.png"
BASE = "checkpoints/exp_sd35_128/step_120000/model.safetensors"
ADAPTER = "runs/iam_lora/final"
STYLE = "runs/htr_ab/real/lines/0000109.png"  # a real IAM writer's hand
TEXTS = [
    "Sentence Database",
    "Sir Roy's United Federal Party is",
    "majority in Northern Rhodesia , but the",
]
CFGS = [1.0, 1.5, 2.0, 3.0, 5.0]  # 1.0 = guidance off; 5.0 = what synth used

device = "cuda"
torch.set_float32_matmul_precision("high")
cfg = Config()
cfg.cond.glyph_line = True
cfg.cond.style_in_context = False
cfg.data.line_height = 128
model = Diffu(cfg).to(device).eval()
load_checkpoint(model, BASE)
load_lora(model, ADAPTER)
ds = cfg.vae.downscale_factor
latent_h = 128 // ds
style = load_style(STYLE).unsqueeze(0).to(device)


def gen(text: str, cs: float, *, steps: int = 24, width_scale: float = 1.0) -> Image.Image:
    w = int(model.guidance_renderer.natural_width(text) * width_scale)
    latent_w = _round_up(w, 16) // ds
    with torch.inference_mode():
        out = model.generate([text], style, latent_hw=(latent_h, latent_w), num_steps=steps, cfg_scale=cs)
    return to_pil(out[0])


def fnt(sz: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


# generate every cell (+ one tight-width, cfg=3 variant per text to test the width hypothesis)
rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
for t in TEXTS:
    cells = [(f"cfg {c}", gen(t, c)) for c in CFGS]
    cells.append(("cfg 3 · width×0.8", gen(t, 3.0, width_scale=0.8)))
    rows.append((t, cells))
    print(f"[done] {t[:30]!r}", flush=True)

# compose
H = 50
LBL = 132
MARGIN = 16
f_t = fnt(16)
f_l = fnt(12)


def sc(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    return im.resize((int(im.width * H / im.height), H), Image.LANCZOS)


scaled = [(t, [(lab, sc(im)) for lab, im in cells]) for t, cells in rows]
line_w = max(im.width for _, cells in scaled for _, im in cells)
W = MARGIN + LBL + line_w + MARGIN
block_h = 22 + (H + 6) * (len(CFGS) + 1) + 18
Hpx = 40 + block_h * len(scaled)
sheet = Image.new("RGB", (W, Hpx), "white")
dr = ImageDraw.Draw(sheet)
dr.text((MARGIN, 12), "iam_lora CFG/width sweep — does looping collapse at low CFG?", font=fnt(18), fill="#111")
y = 40
for t, cells in scaled:
    dr.text((MARGIN, y), f"“{t}”", font=f_t, fill="#222")
    y += 22
    for lab, im in cells:
        dr.text((MARGIN, y + H // 2 - 8), lab, font=f_l, fill="#2563eb")
        sheet.paste(im, (MARGIN + LBL, y))
        y += H + 6
    y += 18
    dr.line([(MARGIN, y - 9), (W - MARGIN, y - 9)], fill="#eee")

sheet.save(OUT)
print("SAVED", OUT, sheet.size, os.path.getsize(OUT), "bytes")
