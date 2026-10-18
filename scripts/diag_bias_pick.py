"""Pick the width bias that restores clean rendering at 128px, verified on Swedish + English.

Generates via the canonical path (biased width + ink_crop) at candidate biases and composes with the real
crop, so we can set _WIDTH_BIAS from evidence rather than theory.
"""

from __future__ import annotations

import os

import torch
from PIL import Image, ImageDraw, ImageFont

from diffu.config import Config
from diffu.generate import _round_up, ink_crop, load_checkpoint, load_style, to_pil
from diffu.model import Diffu

OUT = "/tmp/claude-1000/-home-blackwell-Desktop-diffu/6245d0dc-db0d-4483-891b-dddc762c7c70/scratchpad/bias_pick.png"
BASE = "checkpoints/exp_sd35_128/step_120000/model.safetensors"
CASES = [
    ("Årliga rentan aff Borgo", "data_out_lejonet_128/lines/000000_000.png", "data_out_lejonet_128/lines/000000_000.png"),
    ("Sentence Database", "runs/htr_ab/real/lines/0001069.png", "runs/htr_ab/real/lines/0000109.png"),
    ("Sir Roy's United Federal Party is", "runs/htr_ab/real/lines/0000109.png", "runs/htr_ab/real/lines/0000109.png"),
]
BIASES = [0.45, 0.35, 0.28]

device = "cuda"
torch.set_float32_matmul_precision("high")
cfg = Config()
cfg.cond.glyph_line = True
cfg.cond.style_in_context = False
cfg.data.line_height = 128
model = Diffu(cfg).to(device).eval()
load_checkpoint(model, BASE)
ds = cfg.vae.downscale_factor
latent_h = 128 // ds


def gen(text: str, style: torch.Tensor, bias: float) -> Image.Image:
    raw = model.guidance_renderer.natural_width(text)
    px = min(cfg.data.max_line_width, _round_up(max(128, int(raw * bias)), 16))
    latent_w = px // ds
    with torch.inference_mode():
        out = model.generate([text], style, latent_hw=(latent_h, latent_w), num_steps=24, cfg_scale=5.0)
    return ink_crop(to_pil(out[0]))  # canonical path trims the blank tail


def fnt(sz: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


H, LBL, MARGIN = 50, 160, 16


def sc(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    return im.resize((int(im.width * H / im.height), H), Image.LANCZOS)


rows = []
for text, real_path, style_path in CASES:
    style = load_style(style_path).unsqueeze(0).to(device)
    raw = model.guidance_renderer.natural_width(text)
    cells = [("real", sc(Image.open(real_path)))]
    for b in BIASES:
        cells.append((f"bias {b} (~{int(raw * b / max(len(text), 1))}px/ch)", sc(gen(text, style, b))))
    rows.append((text, cells))
    print(f"[done] {text!r} raw={raw}", flush=True)

line_w = max(im.width for _, cells in rows for _, im in cells)
W = MARGIN + LBL + line_w + MARGIN
block_h = 22 + (H + 6) * (len(BIASES) + 1) + 18
Hpx = 44 + block_h * len(rows)
sheet = Image.new("RGB", (W, Hpx), "white")
dr = ImageDraw.Draw(sheet)
dr.text((MARGIN, 12), "Width-bias pick — canonical path (biased width + ink_crop), base cfg 5", font=fnt(18), fill="#111")
y = 40
for text, cells in rows:
    dr.text((MARGIN, y), f"“{text}”", font=fnt(15), fill="#222")
    y += 22
    for lab, im in cells:
        dr.text((MARGIN, y + H // 2 - 8), lab, font=fnt(11), fill="#16a34a" if lab == "real" else "#2563eb")
        sheet.paste(im, (MARGIN + LBL, y))
        y += H + 6
    y += 18
    dr.line([(MARGIN, y - 9), (W - MARGIN, y - 9)], fill="#eee")
sheet.save(OUT)
print("SAVED", OUT, sheet.size, os.path.getsize(OUT), "bytes")
