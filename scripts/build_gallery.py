"""Build a self-contained HTML gallery from a synthetic-PAGE manifest, to eyeball the results.

    uv run python scripts/build_gallery.py --manifest synth_pages/pages/manifest.jsonl --out gallery.html

Reads ``{image, template, page_cer, n_lines}`` rows, writes a filterable lazy-loaded thumbnail grid (grouped
by template, CER shown). Images are referenced by path (open the HTML from the repo root).
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

_CSS = """
body{background:#14151a;color:#d7d9e0;font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:20px}
h1{font-size:18px;margin:0 0 4px} .sub{color:#8a8f9c;margin-bottom:14px}
.bar{position:sticky;top:0;background:#14151a;padding:10px 0;border-bottom:1px solid #262832;margin-bottom:14px;z-index:9}
button{background:#20222b;color:#c7cad3;border:1px solid #333;border-radius:14px;padding:5px 12px;margin:3px;cursor:pointer;font-size:12px}
button.on{background:#3b5bdb;border-color:#3b5bdb;color:#fff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}
.card{background:#1b1d24;border:1px solid #262832;border-radius:8px;overflow:hidden}
.card img{width:100%;display:block;background:#fff;aspect-ratio:1654/2339;object-fit:contain}
.cap{padding:6px 8px;font-size:11px;color:#9aa0ad;display:flex;justify-content:space-between}
.tpl{color:#7d92e8} .cer{color:#6bbf6b}
"""

_JS = """
function flt(t,b){document.querySelectorAll('.bar button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
document.querySelectorAll('.card').forEach(c=>{c.style.display=(t==='all'||c.dataset.t===t)?'':'none'})}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="gallery.html")
    ap.add_argument("--title", default="diffu-page — synthetic pages")
    ap.add_argument("--limit", type=int, default=0, help="cap cards (0 = all)")
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    counts = Counter(r["template"] for r in rows)
    cers = [r.get("page_cer", 0) for r in rows]
    mean_cer = sum(cers) / max(1, len(cers))

    btns = f'<button class="on" onclick="flt(\'all\',this)">all ({len(rows)})</button>'
    btns += "".join(
        f"<button onclick=\"flt('{html.escape(t)}',this)\">{html.escape(t)} ({n})</button>"
        for t, n in counts.most_common()
    )
    cards = "".join(
        f'<div class="card" data-t="{html.escape(r["template"])}">'
        f'<img loading="lazy" src="{html.escape(r["image"])}">'
        f'<div class="cap"><span class="tpl">{html.escape(r["template"])}</span>'
        f'<span class="cer">CER {r.get("page_cer", 0):.2f} · {r.get("n_lines", 0)}L</span></div></div>'
        for r in rows
    )
    doc = (
        f"<!doctype html><html><head><meta charset=utf-8><title>{html.escape(args.title)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(args.title)}</h1>"
        f'<div class="sub">{len(rows)} gated pages · mean page-CER {mean_cer:.3f} · {len(counts)} templates</div>'
        f'<div class="bar">{btns}</div><div class="grid">{cards}</div>'
        f"<script>{_JS}</script></body></html>"
    )
    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"wrote {args.out}  ({len(rows)} pages, mean CER {mean_cer:.3f}, templates {dict(counts)})")


if __name__ == "__main__":
    main()
