"""Evolution gallery — watch a run's handwriting sharpen across checkpoints.

Generates the SAME texts + writer style from every ``step_*`` checkpoint of a run (cfg=5; the architecture
from the run's ``config.json``, arch flags override it) and assembles both an animated GIF (frames =
checkpoints) and a static grid PNG (rows = texts, cols = checkpoints), so you can see — with your own
eyes — whether the model is still improving or has plateaued. Each checkpoint contributes its EMA weights
(``ema_model.safetensors``), or the raw ``model.safetensors`` of a run saved before training wrote them.

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run python scripts/evolution_gallery.py \
        --ckpt-dir runs/<run> --style <a line crop of the target hand>.png

A ``--style-canvas`` run places the ``--style`` line left of each generated line; give its transcription
with ``--style-text``, else the model sees the hand but not which letters it shows.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

from diffu.config import add_arch_args, apply_arch_args
from diffu.generate import (
    assert_conditioner_loaded,
    build_model,
    generate_line,
    load_checkpoint,
    load_config,
    load_ref_line,
    load_style,
)

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


def _weights(step_dir: Path) -> Path | None:
    """A step dir's EMA weights (what inference uses), else its raw ``model.safetensors`` (older runs)."""
    for name in ("ema_model.safetensors", "model.safetensors"):
        if (step_dir / name).is_file():
            return step_dir / name
    return None


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
    ap.add_argument("--ckpt-dir", required=True, help="a run's out dir (holds its step_<N>/ checkpoints)")
    ap.add_argument("--style", required=True, help="a line crop of the target hand")
    ap.add_argument("--style-text", default="", help="--style-canvas: transcription of the --style line")
    ap.add_argument("--cfg-scale", type=float, default=5.0)
    ap.add_argument("--out", default="/tmp/evolution")
    add_arch_args(ap)
    args = ap.parse_args()

    found = (_weights(d) for d in Path(args.ckpt_dir).glob("step_*"))
    ckpts = sorted((w for w in found if w is not None), key=_step)
    if not ckpts:
        raise SystemExit(f"no step_*/ema_model.safetensors or step_*/model.safetensors under {args.ckpt_dir}")

    cfg = apply_arch_args(load_config(ckpts[-1]), args)
    cell_h = cfg.data.line_height  # cells sized for the run's native line height
    cell_w = cell_h * 6  # ~6:1, the old 380×64 cell aspect
    # Unfused Q/K/V: every checkpoint is loaded into this one model, and a fused copy would keep the first.
    model = build_model(cfg, fuse_qkv=False)
    if model.canvas:
        style, ref = None, load_ref_line(args.style, model.cfg, text=args.style_text)
    else:
        style, ref = load_style(args.style).unsqueeze(0).to(next(model.parameters()).device), None

    frames: list[Image.Image] = []
    grid_cols: list[list[Image.Image]] = []
    steps: list[int] = []
    for cp in ckpts:
        step = _step(cp)
        load_checkpoint(model, str(cp))
        assert_conditioner_loaded(model, str(cp))
        cells = [
            _fit(
                generate_line(model, t, style, num_steps=24, cfg_scale=args.cfg_scale, ref=ref),
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
