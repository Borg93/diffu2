"""Turn a run's periodic sample grids into an IMAGE-ONLY training-evolution view.

A training run saves a labelled comparison grid every ``--sample-every`` steps to
``<run>/samples/step_<N>.png``: a ``want:`` caption, the REAL target row, the OURS (generated) row, and
the recognizer's ``HTR reads:`` read-back. To watch *only* how the diffusion model's own output sharpens
over training, we want NONE of that scaffolding — no REAL row, no captions, no CER/scores — just the
generated handwriting, packed tight. This script crops the OURS strip out of each grid and builds:

  * ``<name>_evolve.gif`` / ``.mp4`` — one chosen sample line, image-only, one frame per step (watch it
    sharpen / evolve; no text burned in)
  * ``<name>_evolve.png``           — the same strips stacked top(early) -> bottom(late) with ZERO padding,
    so the whole training trajectory reads as one tight column

The crop geometry mirrors ``_compose_sample_grid`` in ``src/diffu/train.py``: each of ``--grid-rows``
entries is a vertical block ``style_h + cap_h + 2*line_h + read_h`` tall, and OURS is the 2nd of the two
image rows in that block (REAL is the 1st). The fixed tag-gutter width / caption height are
``SAMPLE_LABEL_W`` / ``SAMPLE_CAPTION_H``, imported from there.

    uv run python scripts/make_training_animation.py --samples-dir runs/<run>/samples --name <run>

Reusable for any run (point ``--samples-dir`` at its ``samples/`` folder).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# The burned-in grid geometry of _compose_sample_grid: the left REAL/OURS tag gutter, and the height of one
# caption line (want / style-ref / HTR-reads).
from diffu.train import SAMPLE_CAPTION_H, SAMPLE_LABEL_W

_STEP_RE = re.compile(r"step_0*(\d+)\.png$")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def list_frames(samples_dir: Path) -> list[tuple[int, Path]]:
    """All ``step_<N>.png`` under ``samples_dir``, sorted by step number."""
    out: list[tuple[int, Path]] = []
    for p in samples_dir.glob("step_*.png"):
        m = _STEP_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def subsample(frames: list[tuple[int, Path]], max_frames: int) -> list[tuple[int, Path]]:
    """Evenly thin to at most ``max_frames`` (keeps first + last)."""
    if len(frames) <= max_frames:
        return frames
    step = (len(frames) - 1) / (max_frames - 1)
    idx = sorted({round(i * step) for i in range(max_frames)})
    return [frames[i] for i in idx]


def crop_generated(grid: Image.Image, rows: int, entry: int, has_style: bool, has_reads: bool) -> Image.Image:
    """Crop just the OURS (generated) strip for sample ``entry`` out of a labelled grid PNG.

    Reconstructs ``_compose_sample_grid``'s layout: each entry block is ``style_h + cap_h + 2*line_h +
    read_h`` tall and the OURS image is the 2nd of its two image rows. With the grid's outer size and the
    fixed caption heights we can solve for ``line_h`` and slice out exactly that row — no captions, no
    REAL row, no tags.
    """
    w, h = grid.size
    style_h = SAMPLE_CAPTION_H if has_style else 0
    read_h = SAMPLE_CAPTION_H if has_reads else 0
    pair_h = h // rows
    line_h = (pair_h - style_h - SAMPLE_CAPTION_H - read_h) // 2
    if line_h <= 0:
        raise SystemExit(
            f"bad crop geometry: --grid-rows {rows} gives line_h={line_h} on a {w}x{h} grid; "
            "check --grid-rows / --no-style / --no-reads against how the run saved its samples"
        )
    if not 0 <= entry < rows:
        raise SystemExit(f"--entry {entry} out of range for --grid-rows {rows}")
    y0 = entry * pair_h + style_h + SAMPLE_CAPTION_H + line_h  # skip style+want captions, then the REAL row
    return grid.crop((SAMPLE_LABEL_W, y0, w, y0 + line_h)).convert("RGB")


def to_width(img: Image.Image, width: int) -> Image.Image:
    """Resize to ``width`` keeping aspect, forcing an even height (H.264 needs even dims)."""
    h = max(2, round(img.height * width / img.width))
    h += h % 2
    return img.resize((width, h), Image.Resampling.LANCZOS)


def stamp_step(img: Image.Image, step: int) -> Image.Image:
    """Optional tiny ``step N`` corner banner (off by default — the request is image-only)."""
    frame = img.copy()
    draw = ImageDraw.Draw(frame)
    font = _font(max(13, frame.width // 48))
    text = f"step {step:,}"
    box = draw.textbbox((0, 0), text, font=font)
    pad = 4
    draw.rectangle((0, 0, box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return frame


def build_gif(frames: list[Image.Image], out: Path, fps: int) -> None:
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True
    )


def build_mp4(frames: list[Image.Image], out: Path, fps: int) -> bool:
    if shutil.which("ffmpeg") is None:  # not installed: subprocess.run would raise, not return non-zero
        return False
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(frames):
            fr.save(Path(tmp) / f"{i:05d}.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            f"{tmp}/%05d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ]
        return subprocess.run(cmd, capture_output=True).returncode == 0


def stack_tight(strips: list[Image.Image], sep: int) -> Image.Image:
    """Stack equal-width strips vertically with ``sep`` px between them (0 = perfectly tight)."""
    w = strips[0].width
    total_h = sum(s.height for s in strips) + sep * (len(strips) - 1)
    sheet = Image.new("RGB", (w, total_h), (255, 255, 255))
    y = 0
    for s in strips:
        sheet.paste(s, (0, y))
        y += s.height + sep
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-dir", required=True, help="<run>/samples folder of step_<N>.png grids")
    ap.add_argument("--out-dir", default="galleries")
    ap.add_argument("--name", default=None, help="artifact prefix (default: parent run dir name)")
    ap.add_argument("--entry", type=int, default=0, help="which sample line to follow (0 = first)")
    ap.add_argument("--grid-rows", type=int, default=4, help="entries per saved grid (generate_samples n)")
    ap.add_argument("--no-style", action="store_true", help="grids have NO 'style ref wrote' caption")
    ap.add_argument("--no-reads", action="store_true", help="grids have NO 'HTR reads' caption")
    ap.add_argument("--max-frames", type=int, default=160, help="thin the GIF/MP4 to at most this many")
    ap.add_argument("--width", type=int, default=1024, help="output width in px (height auto)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--montage-n", type=int, default=40, help="strips in the tight evolution column")
    ap.add_argument("--montage-sep", type=int, default=0, help="px between strips (0 = no padding)")
    ap.add_argument("--step-labels", action="store_true", help="burn a tiny step number per frame")
    args = ap.parse_args()

    samples = Path(args.samples_dir)
    name = args.name or samples.parent.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = list_frames(samples)
    if not frames:
        raise SystemExit(f"no step_*.png under {samples}")
    print(f"{len(frames)} sample grids, step {frames[0][0]:,} -> {frames[-1][0]:,}")

    def strip(path: Path) -> Image.Image:
        grid = Image.open(path)
        cut = crop_generated(grid, args.grid_rows, args.entry, not args.no_style, not args.no_reads)
        return to_width(cut, args.width)

    picks = subsample(frames, args.max_frames)
    print(f"cropping OURS strips (entry {args.entry}) for {len(picks)} frames at width {args.width}…")
    vid_frames = [strip(p) for _, p in picks]
    if args.step_labels:
        vid_frames = [stamp_step(f, s) for f, (s, _) in zip(vid_frames, picks, strict=True)]

    gif = out_dir / f"{name}_evolve.gif"
    build_gif(vid_frames, gif, args.fps)
    print(f"wrote {gif}  ({gif.stat().st_size // 1024} KB)")

    mp4 = out_dir / f"{name}_evolve.mp4"
    if build_mp4(vid_frames, mp4, args.fps):
        print(f"wrote {mp4}  ({mp4.stat().st_size // 1024} KB)")
    else:
        print("ffmpeg unavailable/failed — skipped mp4")

    montage_picks = subsample(frames, args.montage_n)
    sheet = stack_tight([strip(p) for _, p in montage_picks], args.montage_sep)
    evolve = out_dir / f"{name}_evolve.png"
    sheet.save(evolve)
    print(f"wrote {evolve}  ({sheet.width}x{sheet.height}, {evolve.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
