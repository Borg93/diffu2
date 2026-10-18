"""Diagnostic: which IAM approach (if any) renders cleanly — base / LoRA / full-FT / from-scratch?

Renders the same texts through four weight sets on one model instance (hot-swapping the checkpoint, so
the frozen VAE/DINO are built once). If from-scratch (60k IAM-native steps) renders cleanly while the
LoRA loops, the fix is "train longer/better". If everything loops, IAM is beyond this model's conditioning.
"""

from __future__ import annotations

import os

import torch
from PIL import Image, ImageDraw, ImageFont

from diffu.config import Config
from diffu.generate import _round_up, load_checkpoint, load_style, to_pil
from diffu.lora import load_lora
from diffu.model import Diffu

OUT = "/tmp/claude-1000/-home-blackwell-Desktop-diffu/6245d0dc-db0d-4483-891b-dddc762c7c70/scratchpad/model_compare.png"
BASE = "checkpoints/exp_sd35_128/step_120000/model.safetensors"
STYLE = "runs/htr_ab/real/lines/0000109.png"
TEXTS = ["Sentence Database", "Sir Roy's United Federal Party is", "majority in Northern Rhodesia , but the"]
CFG = 3.0

device = "cuda"
torch.set_float32_matmul_precision("high")
cfg = Config()
cfg.cond.glyph_line = True
cfg.cond.style_in_context = False
cfg.data.line_height = 128
model = Diffu(cfg).to(device).eval()
ds = cfg.vae.downscale_factor
latent_h = 128 // ds
style = load_style(STYLE).unsqueeze(0).to(device)


def gen(text: str) -> Image.Image:
    latent_w = _round_up(int(model.guidance_renderer.natural_width(text)), 16) // ds
    with torch.inference_mode():
        out = model.generate([text], style, latent_hw=(latent_h, latent_w), num_steps=24, cfg_scale=CFG)
    return to_pil(out[0])


# full-checkpoint variants first (each load_checkpoint overwrites the backbone), LoRA last.
variants: list[tuple[str, str, str | None]] = [
    ("base (Swedish, no IAM)", BASE, None),
    ("full-FT (6k)", "runs/iam_full/final/model.safetensors", None),
    ("from-scratch (60k)", "runs/iam_scratch/final/model.safetensors", None),
    ("LoRA (4k adapter)", BASE, "runs/iam_lora/final"),
]

cells: dict[tuple[str, str], Image.Image] = {}
for label, ckpt, adapter in variants:
    load_checkpoint(model, ckpt)
    if adapter:
        load_lora(model, adapter)
    for t in TEXTS:
        cells[(t, label)] = gen(t)
    print(f"[done] {label}", flush=True)


def fnt(sz: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


H, LBL, MARGIN = 50, 168, 16
f_t, f_l = fnt(16), fnt(12)


def sc(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    return im.resize((int(im.width * H / im.height), H), Image.LANCZOS)


labels = [v[0] for v in variants]
scaled = {k: sc(v) for k, v in cells.items()}
line_w = max(im.width for im in scaled.values())
W = MARGIN + LBL + line_w + MARGIN
block_h = 22 + (H + 6) * len(labels) + 18
Hpx = 40 + block_h * len(TEXTS)
sheet = Image.new("RGB", (W, Hpx), "white")
dr = ImageDraw.Draw(sheet)
dr.text((MARGIN, 12), "IAM render by approach — base vs LoRA vs full-FT vs from-scratch (cfg 3)", font=fnt(18), fill="#111")
y = 40
for t in TEXTS:
    dr.text((MARGIN, y), f"“{t}”", font=f_t, fill="#222")
    y += 22
    for lab in labels:
        dr.text((MARGIN, y + H // 2 - 8), lab, font=f_l, fill="#2563eb")
        sheet.paste(scaled[(t, lab)], (MARGIN + LBL, y))
        y += H + 6
    y += 18
    dr.line([(MARGIN, y - 9), (W - MARGIN, y - 9)], fill="#eee")

sheet.save(OUT)
print("SAVED", OUT, sheet.size, os.path.getsize(OUT), "bytes")
