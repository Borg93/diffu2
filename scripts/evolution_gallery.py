"""Evolution gallery — watch a run's handwriting sharpen across checkpoints.

Generates the SAME texts + writer style from every ``step_*`` checkpoint of a run (cfg=5, the
correct trained config) and assembles both an animated GIF (frames = checkpoints) and a static grid
PNG (rows = texts, cols = checkpoints), so you can see — with your own eyes — whether the model is
still improving or has plateaued.

    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 uv run python scripts/evolution_gallery.py \
        --ckpt-dir checkpoints/exp_sd35_128 --style data_out_lejonet_128/lines/003425_015.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

from diffu.config import ConditioningConfig, Config
from diffu.generate import generate_line, load_checkpoint, load_style
from diffu.model import Diffu

TEXTS = [
    "Deßes sampt och i synnerheet",
    "om Rijkzens rätt; Rådhe honom",
    "Styrckia Ständerne och Almogen",
    "Kongl. Maj:ts nådige befallning",
]
PAD, HEADER = 6, 22  # cell height tracks the model's line_height (see main); width keeps ~6:1


def _step(path: Path) -> int:
    m = re.search(r"step_(\d+)", str(path))
    if m is None:
        raise ValueError(f"no step_<N> in {path}")
    return int(m.group(1))


def _fit(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    """Fit a generated line onto a white cell_w×cell_h cell (aspect-preserved, left-aligned)."""
    scale = min(cell_w / img.width, cell_h / img.height)
    line = img.convert("RGB").resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    cell = Image.new("RGB", (cell_w, cell_h), "white")
    cell.paste(line, (0, (cell_h - line.height) // 2))
    return cell


def _frame(step: int, cells: list[Image.Image], cell_w: int, cell_h: int) -> Image.Image:
    """One GIF frame: a step header + the N generated lines stacked."""
    frame = Image.new("RGB", (cell_w + 2 * PAD, HEADER + len(cells) * (cell_h + PAD) + PAD), "white")
    ImageDraw.Draw(frame).text((PAD, 4), f"step {step:,}", fill="black")
    for i, c in enumerate(cells):
        frame.paste(c, (PAD, HEADER + i * (cell_h + PAD)))
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Animate a run's handwriting across checkpoints.")
    ap.add_argument("--ckpt-dir", default="checkpoints/exp_sd35_128")
    ap.add_argument("--style", default="data_out_lejonet_128/lines/003425_015.png")
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument(
        "--line-height", type=int, default=None, help="cell height (default: Config().line_height)"
    )
    ap.add_argument("--out", default="/tmp/evolution")
    args = ap.parse_args()

    ckpts = sorted(Path(args.ckpt_dir).glob("step_*/model.safetensors"), key=_step)
    if not ckpts:
        raise SystemExit(f"no step_*/model.safetensors under {args.ckpt_dir}")

    cfg = Config(cond=ConditioningConfig(glyph_line=True, style_in_context=False))  # MATCH training
    cell_h = args.line_height or cfg.data.line_height  # cells sized for the run's native line height
    cell_w = cell_h * 6  # ~6:1, the old 380×64 cell aspect
    model = Diffu(cfg).to("cuda").eval()
    style = load_style(args.style).unsqueeze(0).to("cuda")

    frames: list[Image.Image] = []
    grid_cols: list[list[Image.Image]] = []
    steps: list[int] = []
    for cp in ckpts:
        step = _step(cp)
        load_checkpoint(model, str(cp))
        cells = [
            _fit(
                generate_line(model, t, style, cfg=cfg, num_steps=24, cfg_scale=args.cfg_scale),
                cell_w,
                cell_h,
            )
            for t in TEXTS
        ]
        frames.append(_frame(step, cells, cell_w, cell_h))
        grid_cols.append(cells)
        steps.append(step)
        print(f"step {step}: generated {len(cells)} lines", flush=True)

    frames[0].save(f"{args.out}.gif", save_all=True, append_images=frames[1:], duration=650, loop=0)
    grid = Image.new(
        "RGB",
        (len(steps) * (cell_w + PAD) + PAD, HEADER + len(TEXTS) * (cell_h + PAD) + PAD),
        "white",
    )
    d = ImageDraw.Draw(grid)
    for c, (step, cells) in enumerate(zip(steps, grid_cols, strict=True)):
        x = PAD + c * (cell_w + PAD)
        d.text((x, 4), f"{step // 1000}k", fill="black")
        for r, cell in enumerate(cells):
            grid.paste(cell, (x, HEADER + r * (cell_h + PAD)))
    grid.save(f"{args.out}_grid.png")
    print(f"wrote {args.out}.gif ({len(frames)} frames) + {args.out}_grid.png")


if __name__ == "__main__":
    main()
