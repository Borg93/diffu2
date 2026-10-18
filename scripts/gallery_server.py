"""Tiny dynamic gallery server for browsing sampled line crops over SSH.

Unlike the self-contained ``sample_gallery_html.py`` (one frozen seed), this serves a LIVE page with a
🎲 button + seed/count/collection controls, so you can reshuffle to a new random sample without
re-running anything. Images lazy-load via a ``/thumb`` endpoint (downscaled JPEG), so the page stays
light even over a forwarded port. Stdlib only — no Flask.

    uv run python scripts/gallery_server.py                 # localhost:8000, manifest=data_out_lejonet
    ssh -L 8000:localhost:8000 blackwell@dmlpai01          # then open http://localhost:8000

The manifest is re-read automatically when it changes on disk (so a running prep job's new crops show
up on the next page load). It also serves the static PNG galleries in ``galleries/``.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from PIL import Image

_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.4 system-ui, sans-serif; margin: 0; padding: 16px; }
h1 { font-size: 18px; margin: 0 0 8px; }
.bar { position: sticky; top: 0; background: Canvas; padding: 8px 0 12px; border-bottom: 1px solid #8884;
       margin-bottom: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.bar input { font: inherit; padding: 3px 6px; }
.bar input[type=number] { width: 6em; } .bar input[name=collections] { width: 16em; }
.bar button, .bar a.btn { font: inherit; padding: 4px 12px; border: 1px solid #8886; border-radius: 6px;
       background: #8881; cursor: pointer; text-decoration: none; color: inherit; }
.bar .dice { font-size: 15px; font-weight: 600; }
.sub { color: #888; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr)); gap: 12px; }
figure { margin: 0; border: 1px solid #8884; border-radius: 8px; padding: 8px; background: #80808014; }
figure img { display: block; max-width: 100%; height: auto; background: #fff; border-radius: 4px; min-height: 24px; }
figcaption { margin-top: 6px; word-break: break-word; }
.meta { display: block; margin-top: 3px; font-size: 11px; color: #888; }
"""

# Module-level config + manifest cache (set in main; the handler reads them).
MANIFEST = Path("data_out_lejonet/manifest.jsonl")
LINES_DIR = Path("data_out_lejonet/lines").resolve()
STATIC_DIR = Path("galleries").resolve()
_CACHED_ROWS: list[dict[str, str]] = []
_CACHED_MTIME: float | None = None


def _rows() -> list[dict[str, str]]:
    """Manifest rows, reloaded only when the file's mtime changes (tolerates a half-written last line)."""
    global _CACHED_ROWS, _CACHED_MTIME  # noqa: PLW0603 — small module-level cache shared by handler threads
    try:
        mtime = MANIFEST.stat().st_mtime
    except OSError:
        return []
    if mtime != _CACHED_MTIME:
        rows: list[dict[str, str]] = []
        with MANIFEST.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        _CACHED_ROWS, _CACHED_MTIME = rows, mtime
    return _CACHED_ROWS


def _sample(seed: int, n: int, collections: str) -> tuple[list[dict[str, str]], int]:
    rows = _rows()
    if collections:
        wanted = [c.strip().lower() for c in collections.split(",") if c.strip()]
        rows = [r for r in rows if any(w in str(r.get("collection", "")).lower() for w in wanted)]
    pool = len(rows)
    idx = list(range(pool))
    random.Random(seed).shuffle(idx)
    return [rows[i] for i in idx[:n]], pool


def _page(seed: int, n: int, collections: str, max_width: int) -> bytes:
    sample, pool = _sample(seed, n, collections)
    cells = []
    for r in sample:
        src = f"/thumb?p={quote(r['image'])}&w={max_width}"
        text = html.escape(r.get("text", ""))
        meta = html.escape(f"{r.get('collection', '')} · {r.get('volume', '')}")
        cells.append(f'<figure><img loading="lazy" src="{src}" alt="">'
                     f'<figcaption>{text}<span class="meta">{meta}</span></figcaption></figure>')
    coll_val = html.escape(collections)
    bar = (
        f"<form class='bar' method='get' action='/'>"
        f"<h1>GT-Lejonet sample</h1>"
        f"<button class='dice' type='button' onclick=\"reseed()\">🎲 New random sample</button>"
        f"<label>seed <input type='number' name='seed' value='{seed}'></label>"
        f"<label>count <input type='number' name='n' value='{n}'></label>"
        f"<label>collections <input type='text' name='collections' value='{coll_val}' placeholder='all'></label>"
        f"<button type='submit'>Show</button>"
        f"<a class='btn' href='/reload'>↻ reload data</a>"
        f"<span class='sub'>{len(sample)} of {pool:,} lines · seed {seed}</span>"
        f"</form>"
    )
    script = ("<script>function reseed(){const u=new URL(location.href);"
              "u.searchParams.set('seed',Math.floor(Math.random()*1e9));location.href=u}</script>")
    doc = (f"<!doctype html><html lang='sv'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>GT-Lejonet sample (seed {seed})</title><style>{_CSS}</style></head>"
           f"<body>{bar}<div class='grid'>{''.join(cells)}</div>{script}</body></html>")
    return doc.encode("utf-8")


def _thumb(path: str, max_width: int) -> bytes | None:
    """Read a crop, downscale, return JPEG bytes. Path is sandboxed to LINES_DIR (no arbitrary read)."""
    p = Path(path).resolve()
    if not str(p).startswith(str(LINES_DIR) + os.sep):
        return None
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.width > max_width:
                im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
    except OSError:
        return None
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — match stdlib signature; quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            seed = int(q.get("seed", ["42"])[0] or 42)
            n = max(1, min(2000, int(q.get("n", ["200"])[0] or 200)))
            collections = q.get("collections", [""])[0]
            self._send(_page(seed, n, collections, 700), "text/html; charset=utf-8")
        elif u.path == "/thumb":
            body = _thumb(q.get("p", [""])[0], int(q.get("w", ["700"])[0] or 700))
            if body is None:
                self._send(b"not found", "text/plain", 404)
            else:
                self._send(body, "image/jpeg")
        elif u.path == "/reload":
            global _CACHED_MTIME  # noqa: PLW0603
            _CACHED_MTIME = None  # force re-read on next sample
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
        else:  # static PNG galleries etc., sandboxed to STATIC_DIR
            p = (STATIC_DIR / u.path.lstrip("/")).resolve()
            if str(p).startswith(str(STATIC_DIR)) and p.is_file():
                ctype = "image/png" if p.suffix == ".png" else "application/octet-stream"
                self._send(p.read_bytes(), ctype)
            else:
                self._send(b"not found", "text/plain", 404)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic gallery server with a reshuffle/seed button.")
    ap.add_argument("--manifest", type=Path, default=Path("data_out_lejonet/manifest.jsonl"))
    ap.add_argument("--static-dir", type=Path, default=Path("galleries"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global MANIFEST, LINES_DIR, STATIC_DIR  # noqa: PLW0603 — handler reads these module-level
    MANIFEST = args.manifest
    LINES_DIR = (args.manifest.parent / "lines").resolve()
    STATIC_DIR = args.static_dir.resolve()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving gallery on http://{args.host}:{args.port}  (manifest={MANIFEST})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
