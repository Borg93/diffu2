"""Corpus quality gallery — see WHY low-quality training lines hurt the model.

Scores a sample of real corpus lines two ways — the HTR recognizer's per-line CER (can a SOTA reader even
parse it?) and cheap image heuristics (contrast/ink-coverage/edge-bleed) — diagnoses a concrete reason for
each weak line, and writes a self-contained, image-first HTML gallery (worst-first) you can open in a browser.

It does NOT modify the corpus: it's a diagnostic to eyeball the data the model learns from. A generative model
matches its training distribution, so faded/cropped/mis-transcribed lines teach the model those exact defects.

Run:
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 uv run python scripts/corpus_quality_gallery.py \
        --manifest data_out_lejonet/val.jsonl --n 500 --out galleries/corpus_quality.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import statistics
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from diffu.config import Config
from diffu.data.dataset import load_manifest
from diffu.model.metrics import cer
from diffu.recognizer import load_recognizer, read_lines

_INK_DELTA = 40  # a pixel ≥ this much darker than the paper median counts as ink


class LineStats(BaseModel):
    contrast: float  # paper_median - mean(ink) ; low = faded/washed out
    coverage: float  # fraction of ink pixels ; ~0 = near-blank, high = blotchy/bleed
    top_edge: float  # ink fraction in the top rows (neighbour-line bleed / cropped ascenders)
    bottom_edge: float  # ink in the bottom rows (cropped descenders / bleed)
    side_edge: float  # ink in the left/right columns (horizontally cropped text)


class LineCard(BaseModel):
    uri: str
    gt: str
    read: str
    cer: float
    stats: LineStats
    reasons: list[str]
    severity: str  # bad | weak | clean


def line_stats(img: Image.Image) -> LineStats:
    g = np.asarray(img.convert("L"), dtype=np.int16)
    paper = float(np.median(g))
    ink = g < paper - _INK_DELTA
    cov = float(ink.mean())
    contrast = paper - float(g[ink].mean()) if ink.any() else 0.0
    k = max(1, g.shape[0] // 20)  # ~3px band at height 64
    return LineStats(
        contrast=contrast,
        coverage=cov,
        top_edge=float(ink[:k, :].mean()),
        bottom_edge=float(ink[-k:, :].mean()),
        side_edge=float(max(ink[:, :k].mean(), ink[:, -k:].mean())),
    )


def diagnose(gt: str, read: str, cer_val: float, s: LineStats) -> list[str]:
    """Human-readable reasons a line is low quality + the training harm it causes."""
    out: list[str] = []
    if cer_val >= 0.10:
        out.append(
            f"illegible to the HTR reader (CER {cer_val:.0%}) — degraded scan and/or WRONG transcription → teaches wrong glyph→shape"
        )
    if s.contrast < 45:
        out.append(f"faded / low contrast (paper−ink Δ{s.contrast:.0f}) → model learns washed-out strokes")
    if s.coverage < 0.008:
        out.append("near-blank (almost no ink) → mostly white-paper target, dilutes letter signal")
    if s.coverage > 0.28:
        out.append(f"very heavy ink / blotch (coverage {s.coverage:.0%}) → model learns smudge/bleed")
    if s.top_edge > 0.02 or s.bottom_edge > 0.02:
        out.append(
            "ink touches top/bottom edge → neighbour-line bleed or cropped ascenders/descenders (stray marks)"
        )
    if s.side_edge > 0.06:
        out.append("ink at left/right edge → horizontally cropped word (cut letters)")
    if gt and len(read) < 0.6 * len(gt):
        out.append(f"reader got {len(read)} chars vs {len(gt)} in GT → truncated/missing content or bad GT")
    return out


def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _severity(cer_val: float, reasons: list[str]) -> str:
    if cer_val >= 0.10:
        return "bad"
    if reasons:
        return "weak"
    return "clean"


def render_html(cards: list[LineCard], n: int, manifest: str) -> str:
    cers = [c.cer for c in cards]
    bad = sum(c.severity == "bad" for c in cards)
    weak = sum(c.severity == "weak" for c in cards)
    clean = sum(c.severity == "clean" for c in cards)
    med = statistics.median(cers)
    cells = "\n".join(
        f"""<figure class="card {c.severity}">
  <img src="{c.uri}" alt="line">
  <figcaption>
    <div class="cer {c.severity}">CER {c.cer:.0%}</div>
    <div class="txt"><b>GT:</b> {html.escape(c.gt)}</div>
    <div class="txt read"><b>read:</b> {html.escape(c.read)}</div>
    <ul>{"".join(f"<li>{html.escape(r)}</li>" for r in c.reasons)}</ul>
  </figcaption>
</figure>"""
        for c in cards
    )
    return f"""<!doctype html><meta charset="utf-8"><title>Diffu corpus quality</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0d1117;color:#e6edf3}}
 header{{padding:24px 28px;border-bottom:1px solid #30363d;background:#161b22}}
 h1{{margin:0 0 6px}} .sub{{color:#9da7b3}}
 .why{{margin:14px 0 0;padding:14px 16px;background:#1c2128;border-left:3px solid #d29922;border-radius:6px;max-width:980px}}
 .why b{{color:#f0c674}}
 .stats{{margin-top:12px;display:flex;gap:18px;flex-wrap:wrap}}
 .pill{{padding:4px 10px;border-radius:20px;font-weight:600}}
 .pill.bad{{background:#5a1e1e;color:#ffb3b3}} .pill.weak{{background:#5a4a1e;color:#ffe3a3}} .pill.clean{{background:#1e5a2e;color:#a3ffb8}}
 main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:16px;padding:20px}}
 .card{{margin:0;background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}}
 .card.bad{{border-color:#7d2a2a}} .card.weak{{border-color:#7d6a2a}}
 .card img{{display:block;width:100%;background:#fff;image-rendering:auto;border-bottom:1px solid #30363d}}
 figcaption{{padding:10px 12px}}
 .cer{{display:inline-block;font-weight:700;padding:2px 8px;border-radius:4px;margin-bottom:6px}}
 .cer.bad{{background:#5a1e1e;color:#ffb3b3}} .cer.weak{{background:#5a4a1e;color:#ffe3a3}} .cer.clean{{background:#1e5a2e;color:#a3ffb8}}
 .txt{{font-size:13px;word-break:break-word}} .read{{color:#9da7b3}}
 ul{{margin:8px 0 0;padding-left:18px;color:#ffd9a3;font-size:12.5px}} li{{margin:2px 0}}
</style>
<header>
 <h1>Diffu corpus quality — {n} lines from <code>{html.escape(manifest)}</code></h1>
 <div class="sub">Sorted worst-first. Scored by the HTR recognizer (per-line CER) + image heuristics. The corpus is real, aged historical Swedish handwriting.</div>
 <div class="stats">
   <span class="pill bad">{bad} bad (CER ≥ 10%)</span>
   <span class="pill weak">{weak} weak (faded / cropped / bleed)</span>
   <span class="pill clean">{clean} clean</span>
   <span class="pill" style="background:#21262d;color:#e6edf3">median CER {med:.1%}</span>
 </div>
 <div class="why">
   <b>Why this hurts the model.</b> A diffusion model learns the <b>distribution</b> of its data and cannot become
   crisper than what it sees. Each defect teaches a specific bad behaviour:
   <b>faded</b> lines → washed-out strokes; <b>top/bottom edge ink</b> → stray marks above/below the line;
   <b>wrong transcriptions</b> (GT ≠ image) → the model maps the wrong letters to shapes and renders them
   confidently; <b>cropped</b> lines → cut-off words; <b>blotchy/near-blank</b> → smudge or empty output.
   These are not random noise the model averages out — they are <b>consistent</b> patterns it reproduces,
   directly lowering legibility (gen_CER) and sharpness. The fix is curation (up-weight the clean lines,
   down-weight/repair the worst), not architecture.
 </div>
</header>
<main>
{cells}
</main>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an HTML gallery of corpus quality (worst lines + why).")
    ap.add_argument("--manifest", default="data_out_lejonet/val.jsonl")
    ap.add_argument("--n", type=int, default=500, help="lines to score")
    ap.add_argument("--out", default="galleries/corpus_quality.html")
    ap.add_argument("--htr", default=Config().aux.htr_recognizer_eval)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)[: args.n]
    imgs: list[Image.Image] = []
    for r in rows:
        with Image.open(r["image"]) as im:
            imgs.append(im.convert("RGB"))
    proc, model = load_recognizer(args.htr, args.device)
    reads = read_lines(proc, model, imgs)

    cards: list[LineCard] = []
    for r, img, read in zip(rows, imgs, reads, strict=True):
        gt = str(r["text"])
        c = cer([read], [gt])
        st = line_stats(img)
        reasons = diagnose(gt, read, c, st)
        cards.append(
            LineCard(
                uri=to_data_uri(img),
                gt=gt,
                read=read,
                cer=c,
                stats=st,
                reasons=reasons,
                severity=_severity(c, reasons),
            )
        )
    cards.sort(key=lambda c: c.cer, reverse=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(cards, len(cards), args.manifest), encoding="utf-8")
    bad = sum(c.severity == "bad" for c in cards)
    weak = sum(c.severity == "weak" for c in cards)
    print(f"wrote {out}  ({len(cards)} lines: {bad} bad, {weak} weak, {len(cards) - bad - weak} clean)")


if __name__ == "__main__":
    main()
