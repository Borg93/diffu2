"""Control: does the BASE model render its NATIVE Swedish cleanly, or does it loop like it does on English?

If Swedish is clean at cfg 5/3, the looping is an out-of-domain (English/IAM) failure and the released
model is fine in-domain. If Swedish loops too, the inference path (or the model) is the problem.
"""

from __future__ import annotations

import json
import os

import torch
from PIL import Image, ImageDraw, ImageFont

from diffu.config import Config
from diffu.generate import _round_up, load_checkpoint, load_style, to_pil
from diffu.model import Diffu

OUT = "/tmp/claude-1000/-home-blackwell-Desktop-diffu/6245d0dc-db0d-4483-891b-dddc762c7c70/scratchpad/swedish_control.png"
BASE = "checkpoints/exp_sd35_128/step_120000/model.safetensors"
MANIFEST = "data_out_lejonet_128/train.jsonl"

with open(MANIFEST, encoding="utf-8") as _f:
    rows_all = [json.loads(x) for x in _f]
picks = [r for r in rows_all if 14 <= len(r["text"]) <= 40][:3]
STYLE = picks[0]["image"]  # a real Swedish hand as style ref

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
style = load_style(STYLE).unsqueeze(0).to(device)


def gen(text: str, cs: float) -> Image.Image:
    latent_w = _round_up(int(model.guidance_renderer.natural_width(text)), 16) // ds
    with torch.inference_mode():
        out = model.generate([text], style, latent_hw=(latent_h, latent_w), num_steps=24, cfg_scale=cs)
    return to_pil(out[0])


def fnt(sz: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


H, LBL, MARGIN = 50, 130, 16
variants = ["real", "gen cfg 5", "gen cfg 3"]


def sc(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    return im.resize((int(im.width * H / im.height), H), Image.LANCZOS)


cells: dict[tuple[int, str], Image.Image] = {}
for i, r in enumerate(picks):
    cells[(i, "real")] = sc(Image.open(r["image"]))
    cells[(i, "gen cfg 5")] = sc(gen(r["text"], 5.0))
    cells[(i, "gen cfg 3")] = sc(gen(r["text"], 3.0))
    print(f"[done] {r['text']!r}", flush=True)

line_w = max(im.width for im in cells.values())
W = MARGIN + LBL + line_w + MARGIN
block_h = 22 + (H + 6) * len(variants) + 18
Hpx = 40 + block_h * len(picks)
sheet = Image.new("RGB", (W, Hpx), "white")
dr = ImageDraw.Draw(sheet)
dr.text((MARGIN, 12), "Swedish control — BASE model on its native domain (GT-Lejonet)", font=fnt(18), fill="#111")
y = 40
for i, r in enumerate(picks):
    dr.text((MARGIN, y), f"“{r['text']}”", font=fnt(16), fill="#222")
    y += 22
    for v in variants:
        dr.text((MARGIN, y + H // 2 - 8), v, font=fnt(12), fill="#16a34a" if v == "real" else "#2563eb")
        sheet.paste(cells[(i, v)], (MARGIN + LBL, y))
        y += H + 6
    y += 18
    dr.line([(MARGIN, y - 9), (W - MARGIN, y - 9)], fill="#eee")

sheet.save(OUT)
print("SAVED", OUT, sheet.size, os.path.getsize(OUT), "bytes")
