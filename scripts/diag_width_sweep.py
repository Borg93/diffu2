"""Decisive test: is the looping a WIDTH bug (natural_width over-predicts -> model fills by looping)?

For a Swedish and an English text, render the BASE model at several latent widths — natural_width and
fractions of it, plus the REAL crop's exact width — and compose with the real line. If the looping
collapses to clean handwriting as the width tightens to the real width, the failure is width, not the model.
"""

from __future__ import annotations

import os

import torch
from PIL import Image, ImageDraw, ImageFont

from diffu.config import Config
from diffu.generate import _round_up, load_checkpoint, load_style, to_pil
from diffu.model import Diffu

OUT = "/tmp/claude-1000/-home-blackwell-Desktop-diffu/6245d0dc-db0d-4483-891b-dddc762c7c70/scratchpad/width_sweep.png"
BASE = "checkpoints/exp_sd35_128/step_120000/model.safetensors"
# (text, real crop, style ref)
CASES = [
    ("Årliga rentan aff Borgo", "data_out_lejonet_128/lines/000000_000.png", "data_out_lejonet_128/lines/000000_000.png"),
    ("Sentence Database", "runs/htr_ab/real/lines/0001069.png", "runs/htr_ab/real/lines/0000109.png"),
]

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


def gen(text: str, style: torch.Tensor, px_w: int) -> Image.Image:
    latent_w = _round_up(int(px_w), 16) // ds
    with torch.inference_mode():
        out = model.generate([text], style, latent_hw=(latent_h, latent_w), num_steps=24, cfg_scale=5.0)
    return to_pil(out[0])


def fnt(sz: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


H, LBL, MARGIN = 50, 190, 16


def sc(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    return im.resize((int(im.width * H / im.height), H), Image.LANCZOS)


rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
for text, real_path, style_path in CASES:
    style = load_style(style_path).unsqueeze(0).to(device)
    real = Image.open(real_path)
    real_w = real.width
    nat_w = int(model.guidance_renderer.natural_width(text))
    print(f"{text!r}: natural_width={nat_w}px  real_width={real_w}px  ratio={nat_w/real_w:.2f}x", flush=True)
    cells = [(f"real ({real_w}px)", sc(real))]
    for lab, w in [(f"natural {nat_w}px", nat_w), ("×0.6", int(nat_w * 0.6)),
                   ("×0.45", int(nat_w * 0.45)), (f"real-width {real_w}px", real_w)]:
        cells.append((lab, sc(gen(text, style, w))))
    rows.append((text, cells))
    print(f"[done] {text!r}", flush=True)

line_w = max(im.width for _, cells in rows for _, im in cells)
W = MARGIN + LBL + line_w + MARGIN
block_h = 22 + (H + 6) * 5 + 18
Hpx = 44 + block_h * len(rows)
sheet = Image.new("RGB", (W, Hpx), "white")
dr = ImageDraw.Draw(sheet)
dr.text((MARGIN, 10), "Width sweep — is the looping a natural_width over-prediction? (base, cfg 5)", font=fnt(18), fill="#111")
dr.text((MARGIN, 30), "if looping collapses as width tightens to the real crop, it's a width bug, not the model.", font=fnt(12), fill="#666")
y = 46
for text, cells in rows:
    dr.text((MARGIN, y), f"“{text}”", font=fnt(16), fill="#222")
    y += 22
    for lab, im in cells:
        green = lab.startswith("real")
        dr.text((MARGIN, y + H // 2 - 8), lab, font=fnt(11), fill="#16a34a" if green else "#2563eb")
        sheet.paste(im, (MARGIN + LBL, y))
        y += H + 6
    y += 18
    dr.line([(MARGIN, y - 9), (W - MARGIN, y - 9)], fill="#eee")

sheet.save(OUT)
print("SAVED", OUT, sheet.size, os.path.getsize(OUT), "bytes")
