"""Export a trackio run's logged samples as an IMPROVEMENT MATRIX (a timeline grid).

Frames are laid out in *reading order* so training progresses top-left → bottom-right:
the earliest step sits in the top-left cell, the latest/best in the bottom-right. Two styles:

  --style image      (default) a clean grid of just the GENERATED handwriting — the OURS
                     rows are cropped out of each logged strip, no want:/HTR-read clutter.
                     This is the "watch it sharpen" figure.
  --style annotated  the full logged strips (want: / REAL / OURS / HTR reads) tiled into the
                     same matrix — the diagnostic "what does the recognizer read it as" gallery.

    uv run python scripts/gallery.py --run brave-forest-1
    uv run python scripts/gallery.py --run brave-forest-1 --style annotated
    uv run python scripts/gallery.py --run brave-forest-1 --line all --with-real --cols 3

Reads the trackio SQLite DB for a run's media (the REAL/OURS sample strips logged every
``--sample-every`` steps). Handles the trackio run-name collision (two runs sharing an
auto-name) by keeping the run_id with the most logged frames. Long timelines are evenly
subsampled to ``--max-frames`` so the first and last step are always shown.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

PILImage = Image.Image

# Geometry of a logged strip, mirrored from train.py:_compose_sample_grid. The left label
# column and the caption band heights are fixed; the per-line height (line_h) is derived per
# strip from its pixel height, so this stays correct even if line_height changes between runs.
LBL_W, CAP_H = 88, 22


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resolve(file_path: str, trackio_dir: Path) -> Path | None:
    """Resolve a trackio media ``file_path`` (e.g. 'diffu/run/500/uuid.png') to a file on disk."""
    for cand in (trackio_dir / file_path, trackio_dir / "media" / file_path):
        if cand.is_file():
            return cand
    matches = list(trackio_dir.rglob(Path(file_path).name))  # fall back: search by basename
    return matches[0] if matches else None


def run_media(db: Path, run: str) -> list[tuple[int, str, str]]:
    """Return ``[(step, file_path, caption)]`` for the run's logged images, step-sorted.

    On a run-name collision (trackio auto-named two runs identically) keep only the ``run_id``
    with the most media rows — the real run, not a killed stub.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT run_id, step, json_extract(metrics, '$.samples') AS s "
            "FROM metrics WHERE run_name = ? AND s IS NOT NULL ORDER BY step",
            (run,),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return []
    counts: dict[str, int] = {}
    for run_id, _, _ in rows:
        counts[run_id] = counts.get(run_id, 0) + 1
    keep = max(counts, key=counts.__getitem__)
    out: list[tuple[int, str, str]] = []
    for run_id, step, payload in rows:
        if run_id != keep:
            continue
        img = json.loads(payload)
        out.append((int(step), img["file_path"], img.get("caption", "")))
    return out


def pick_run(db: Path) -> str:
    """Return the run_name with the most logged sample frames (the longest real run)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT run_name, COUNT(*) c FROM metrics "
            "WHERE json_extract(metrics, '$.samples') IS NOT NULL GROUP BY run_name ORDER BY c DESC"
        ).fetchall()
    finally:
        con.close()
    if not rows:
        raise SystemExit(f"no runs with logged samples found in {db}")
    return rows[0][0]


def subsample(media: list[tuple[int, str, str]], stride: int, max_frames: int) -> list[tuple[int, str, str]]:
    """Thin a long timeline. ``stride`` keeps every Nth frame; ``max_frames`` then evenly samples
    down to that count while always keeping the first and last step."""
    if stride > 1:
        media = media[::stride]
    if max_frames and len(media) > max_frames:
        last = len(media) - 1
        idx = sorted({round(i * last / (max_frames - 1)) for i in range(max_frames)})
        media = [media[i] for i in idx]
    return media


def _strip_geometry(caption: str, im: PILImage) -> tuple[int, int, int]:
    """Derive ``(n_lines, line_h, pair_h)`` for a logged strip from its caption and pixel height.

    Each line occupies ``pair_h = CAP_H + 2*line_h + read_h`` where ``read_h == CAP_H`` only when
    the recognizer's reads were burned in (caption contains 'read'). ``n_lines`` is the number of
    ``want ...`` entries in the caption.
    """
    n = max(caption.count("want "), 1)
    read_h = CAP_H if "read" in caption else 0
    pair_h = im.height // n
    line_h = (pair_h - CAP_H - read_h) // 2
    return n, line_h, pair_h


def _crop_cell(im: PILImage, caption: str, line: str, with_real: bool) -> PILImage:
    """Crop the chosen handwriting line(s) out of a strip — OURS only, or REAL stacked above OURS.

    ``line`` is a line index as a string, or ``"all"`` to stack every line. No captions/tags are
    included: only the raw generated (and optionally real) ink rows.
    """
    n, line_h, pair_h = _strip_geometry(caption, im)
    idxs = range(n) if line == "all" else [max(0, min(int(line), n - 1))]
    rows: list[PILImage] = []
    for i in idxs:
        top = i * pair_h + CAP_H
        if with_real:
            rows.append(im.crop((LBL_W, top, im.width, top + line_h)))  # REAL row
        rows.append(im.crop((LBL_W, top + line_h, im.width, top + 2 * line_h)))  # OURS row
    w = im.width - LBL_W
    gut = 2 if with_real else 1
    out = Image.new("RGB", (w, sum(r.height for r in rows) + gut * (len(rows) - 1)), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + gut
    return out


def _ink_right(cell: PILImage, tol: int = 22) -> int:
    """Rightmost column of a cell that still has ink (differs from the blank background).

    Background is sampled from the right-edge midpoint, which is blank for short words. Returns the
    full width when the cell is ink everywhere (e.g. an early all-noise sample).
    """
    bg = cell.getpixel((cell.width - 1, cell.height // 2))
    diff = ImageChops.difference(cell, Image.new(cell.mode, cell.size, bg)).convert("L")
    bbox = diff.point(lambda p: 255 if p > tol else 0).getbbox()
    return bbox[2] if bbox else cell.width


def _trim_width(cells: list[PILImage], quantile: float = 0.9, margin: int = 12) -> int:
    """Shared crop width that kills the dead whitespace right of short words while keeping every
    cell the same width (so the matrix stays aligned). Uses a high quantile of per-cell ink extents
    so a few all-noise early frames don't veto the trim; clamps to the source width."""
    rights = sorted(_ink_right(c) for c in cells)
    w = rights[min(len(rights) - 1, round(quantile * (len(rights) - 1)))]
    return min(cells[0].width, max(60, w + margin))


def _cer(caption: str) -> str:
    m = re.search(r"gen_CER ([0-9.]+)", caption)
    return f"CER {m.group(1)}" if m else ""


def _step_label(step: int) -> str:
    return f"{step / 1000:g}k" if step >= 1000 else str(step)


def build_matrix(
    db: Path,
    run: str,
    out: Path,
    *,
    style: str,
    line: str,
    with_real: bool,
    cols: int,
    cell_w: int,
    stride: int,
    max_frames: int,
    labels: bool,
    trim: bool,
) -> int:
    trackio_dir = db.parent
    media = subsample(run_media(db, run), stride, max_frames)
    if not media:
        raise SystemExit(f"no logged images found for run {run!r} in {db}")

    raw: list[tuple[int, str, PILImage]] = []
    for step, file_path, caption in media:
        path = _resolve(file_path, trackio_dir)
        if path is None:
            print(f"  [skip] step {step}: media file not found ({file_path})")
            continue
        im = Image.open(path).convert("RGB")
        cell = im if style == "annotated" else _crop_cell(im, caption, line, with_real)
        raw.append((step, caption, cell))
    if not raw:
        raise SystemExit("found media rows but none of the files resolved on disk")

    crop_w = _trim_width([c for _, _, c in raw]) if (trim and style == "image") else None
    cells: list[tuple[int, str, PILImage]] = []
    for step, caption, cell in raw:
        if crop_w is not None:
            cell = cell.crop((0, 0, crop_w, cell.height))
        cells.append((step, caption, cell.resize((cell_w, round(cell.height * cell_w / cell.width)))))

    cols = max(1, min(cols, len(cells)))
    rows = math.ceil(len(cells) / cols)
    cell_h = max(c.height for _, _, c in cells)
    lab_h = 16 if labels else 0
    title_h, pad = 30, 5
    font, title_font = _font(12), _font(15)
    grid = "#5b5b5b"

    pitch_x = cell_w + 2 * pad
    pitch_y = cell_h + lab_h + 2 * pad
    canvas_w = cols * pitch_x + 1
    canvas_h = title_h + rows * pitch_y + 1
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    out.parent.mkdir(parents=True, exist_ok=True)
    steps = [s for s, _, _ in cells]
    title = f"{run}  ·  steps {_step_label(min(steps))}→{_step_label(max(steps))}  ·  {len(cells)} frames  ·  training →"
    draw.text((1, 8), title, fill="#111111", font=title_font)

    for i, (step, caption, cell) in enumerate(cells):
        r, c = divmod(i, cols)
        x0, y0 = c * pitch_x, title_h + r * pitch_y
        canvas.paste(cell, (x0 + pad, y0 + pad + (cell_h - cell.height) // 2))
        if labels:
            tag = f"{_step_label(step)}   {_cer(caption)}".strip()
            draw.text((x0 + pad, y0 + pad + cell_h + 1), tag, fill="#444444", font=font)

    for c in range(cols + 1):  # the matrix grid (table lattice) the user asked for
        x = c * pitch_x
        draw.line([(x, title_h), (x, canvas_h - 1)], fill=grid, width=1)
    for r in range(rows + 1):
        y = title_h + r * pitch_y
        draw.line([(0, y), (canvas_w - 1, y)], fill=grid, width=1)

    canvas.save(out)
    return len(cells)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a trackio run's sample grids as an improvement-matrix PNG.")
    ap.add_argument("--run", default=None, help="trackio run name (default: the run with the most frames)")
    ap.add_argument("--project", default="diffu", help="trackio project (selects the .db file)")
    ap.add_argument("--db", type=Path, default=None, help="override the trackio DB path")
    ap.add_argument("--out", type=Path, default=None, help="output PNG (default galleries/gallery_<run>[_<style>].png)")
    ap.add_argument("--out-dir", type=Path, default=Path("galleries"), help="folder for the default output name")
    ap.add_argument(
        "--style", choices=("image", "annotated"), default="image",
        help="'image': clean generated-handwriting grid (default); 'annotated': full want/REAL/OURS/HTR strips",
    )
    ap.add_argument("--line", default="0", help="which held-out line to show in image style: index or 'all' (default 0)")
    ap.add_argument("--with-real", action="store_true", help="image style: stack the REAL target above each OURS row")
    ap.add_argument("--cols", type=int, default=6, help="matrix columns (timeline fills left→right, top→bottom)")
    ap.add_argument("--cell-width", type=int, default=380, help="per-cell width px (lines rescaled to this)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame before subsampling")
    ap.add_argument("--max-frames", type=int, default=24, help="evenly subsample to at most N frames (0 = keep all)")
    ap.add_argument("--no-labels", action="store_true", help="omit the per-cell step·CER labels (pure image grid)")
    ap.add_argument("--no-trim", action="store_true", help="image style: keep full line width (don't trim trailing blank space)")
    args = ap.parse_args()

    db = args.db or Path.home() / ".cache/huggingface/trackio" / f"{args.project}.db"
    run = args.run or pick_run(db)
    suffix = "" if args.style == "image" else f"_{args.style}"
    out = args.out or args.out_dir / f"gallery_{run}{suffix}.png"
    n = build_matrix(
        db, run, out,
        style=args.style, line=args.line, with_real=args.with_real,
        cols=args.cols, cell_w=args.cell_width, stride=args.stride,
        max_frames=args.max_frames, labels=not args.no_labels, trim=not args.no_trim,
    )
    print(f"wrote {out} — {n} frames from run {run!r} ({args.style} style)")


if __name__ == "__main__":
    main()
