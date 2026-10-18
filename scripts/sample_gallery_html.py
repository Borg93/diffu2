"""Build a SELF-CONTAINED HTML gallery of a random sample of line crops + their transcriptions.

Safe to inspect a multi-million-image dataset over SSH: it samples only ``--n`` crops and embeds
them as base64 JPEG into ONE ``.html`` file. ``scp`` that single file and open it locally — there is
no image folder to load, no server to run, nothing to crash a file browser or the SSH session.

    uv run python scripts/sample_gallery_html.py --manifest data_out_lejonet/manifest.jsonl --n 200
    uv run python scripts/sample_gallery_html.py --collections poliskammare,trolldom --n 150
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.4 system-ui, sans-serif; margin: 0; padding: 16px; }
h1 { font-size: 18px; margin: 0 0 4px; }
.sub { color: #888; margin: 0 0 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr)); gap: 12px; }
figure { margin: 0; border: 1px solid #8884; border-radius: 8px; padding: 8px; background: #80808014; }
figure img { display: block; max-width: 100%; height: auto; background: #fff; border-radius: 4px; }
figcaption { margin-top: 6px; font-size: 14px; word-break: break-word; }
.meta { display: block; margin-top: 3px; font-size: 11px; color: #888; }
"""


def load_rows(manifest: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a half-written final line if prep is still running
    return rows


def thumb_data_uri(png_path: Path, max_width: int) -> str | None:
    """Read a line crop, downscale to ``max_width``, return a base64 JPEG data URI (small + portable)."""
    try:
        with Image.open(png_path) as im:
            im = im.convert("RGB")
            if im.width > max_width:
                im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
    except OSError:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build(manifest: Path, out: Path, n: int, max_width: int, collections: str | None, seed: int) -> int:
    rows = load_rows(manifest)
    if collections:
        wanted = [c.strip().lower() for c in collections.split(",") if c.strip()]
        rows = [r for r in rows if any(w in str(r.get("collection", "")).lower() for w in wanted)]
    if not rows:
        raise SystemExit(f"no rows in {manifest} (filters: {collections!r})")

    pool = len(rows)
    random.Random(seed).shuffle(rows)
    cells: list[str] = []
    shown_collections: Counter[str] = Counter()
    for r in rows:
        if len(cells) >= n:
            break
        uri = thumb_data_uri(Path(r["image"]), max_width)
        if uri is None:
            continue
        text = html.escape(r.get("text", ""))
        meta = html.escape(f"{r.get('collection', '')} · {r.get('volume', '')}")
        cells.append(f'<figure><img src="{uri}" loading="lazy" alt=""><figcaption>{text}'
                     f'<span class="meta">{meta}</span></figcaption></figure>')
        shown_collections[str(r.get("collection", ""))] += 1

    sub = (f"{len(cells)} random line crops of {pool:,} in the manifest"
           f"  ·  {len(shown_collections)} collections  ·  seed {seed}")
    doc = (f"<!doctype html><html lang='sv'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>GT-Lejonet sample — {len(cells)} lines</title><style>{_CSS}</style></head>"
           f"<body><h1>GT-Lejonet line-crop sample</h1><p class='sub'>{html.escape(sub)}</p>"
           f"<div class='grid'>{''.join(cells)}</div></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return len(cells)


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-contained HTML gallery of sampled line crops.")
    ap.add_argument("--manifest", type=Path, default=Path("data_out_lejonet/manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("galleries/gt_lejonet_sample.html"))
    ap.add_argument("--n", type=int, default=200, help="number of crops to sample (kept small to stay portable)")
    ap.add_argument("--max-width", type=int, default=700, help="downscale wider crops to this before embedding")
    ap.add_argument("--collections", default=None, help="comma-separated substrings to include (default: all)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (change for a fresh random set)")
    args = ap.parse_args()

    n = build(args.manifest, args.out, args.n, args.max_width, args.collections, args.seed)
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} — {n} crops, {size_mb:.1f} MB (self-contained; scp this one file)")


if __name__ == "__main__":
    main()
