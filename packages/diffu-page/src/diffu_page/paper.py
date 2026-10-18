"""Paper backgrounds for ``compose_page`` — the page to seat generated lines on.

Two sources, one module:

- :func:`paper_background` — the runtime background for a page: a random crop from a real-scan corpus
  (``paper_dir``) if available, else a procedural aged-paper texture.
- :func:`extract_paper_crops` — build that corpus by cropping the *blank* regions out of real document
  scans, so the compositor gets authentic fibre, mottling, faint bleed-through and binding shadows rather
  than clean procedural paper — letting the synthetic pages span the *real* distribution, not just the
  pristine end.

    diffu-paper-crops tests_data /tmp/real_paper --per-scan 2   # then compose_page(..., paper_dir=...)
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import typer
from pydantic import BaseModel

_SCAN_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class PaperTone(BaseModel):
    """A named paper-tone target for a scan profile: the base luminance range of procedural paper plus a
    BGR channel tint that sets its hue (warm sepia, cool grey, yellowed, near-neutral). The tint is also
    multiplied onto a real-scan crop so a profile reads consistently whichever paper source is used."""

    model_config = {"frozen": True}

    base_tone: tuple[int, int] = (214, 242)  # procedural paper luminance range (sampled per page)
    tint: tuple[float, float, float] = (0.90, 1.0, 1.0)  # BGR multiplier -> paper hue


def _apply_tint(paper: np.ndarray, tone: PaperTone) -> np.ndarray:
    return np.clip(paper.astype(np.float32) * np.array(tone.tint, np.float32), 0, 255).astype(np.uint8)


def _load_paper(paper_dir: str, width: int, height: int, rng: random.Random) -> np.ndarray | None:
    """A random real-paper crop from ``paper_dir``, cover-scaled and windowed to ``(height, width)``."""
    files = [p for p in Path(paper_dir).iterdir() if p.suffix.lower() in _SCAN_EXT]
    if not files:
        return None
    img = cv2.imread(str(rng.choice(files)), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max(width / w, height / h)  # cover
    img = cv2.resize(img, (int(w * scale) + 1, int(h * scale) + 1), interpolation=cv2.INTER_AREA)
    y = rng.randint(0, max(0, img.shape[0] - height))
    x = rng.randint(0, max(0, img.shape[1] - width))
    return img[y : y + height, x : x + width].copy()


def paper_background(
    width: int,
    height: int,
    paper_dir: str | None = None,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
    *,
    tone: PaperTone | None = None,
) -> np.ndarray:
    """A page-sized paper background: a random real-scan crop from ``paper_dir`` if given, else procedural
    (light aged tone + low-frequency lighting + soft stains + grain).

    ``tone`` (a scan profile's :class:`PaperTone`) sets the procedural paper's luminance range and hue, and
    tints a real crop the same way so the profile reads consistently. Omitting it is byte-identical to the
    original default paper.
    """
    rng = rng or random.Random()
    np_rng = np_rng or np.random.default_rng()
    if paper_dir:
        paper = _load_paper(paper_dir, width, height, rng)
        if paper is not None:
            return _apply_tint(paper, tone) if tone is not None else paper
    # real archive scans skew aged: ~40% are a darker/heavily-aged stock, the rest a (still warm) lighter
    # sheet — the degraded end synthetic data usually misses by sitting uniformly clean/bright.
    if tone is None:
        base_val = rng.randint(168, 210) if rng.random() < 0.4 else rng.randint(214, 242)
    else:
        base_val = rng.randint(*tone.base_tone)
    base = np.full((height, width, 3), float(base_val), np.float32)
    gx = np.linspace(rng.uniform(-10, 10), rng.uniform(-10, 10), width, dtype=np.float32)
    gy = np.linspace(rng.uniform(-12, 12), rng.uniform(-12, 12), height, dtype=np.float32)
    base += gx[None, :, None] + gy[:, None, None]  # low-frequency lighting
    stain = np.zeros((height, width), np.float32)  # soft aged-paper blotches/stains
    for _ in range(rng.randint(6, 16)):
        cx, cy = rng.randint(0, width), rng.randint(0, height)
        ax, ay = rng.randint(width // 12, width // 4), rng.randint(height // 14, height // 4)
        cv2.ellipse(stain, (cx, cy), (ax, ay), 0, 0, 360, rng.uniform(-14, 8), -1)
    small = cv2.resize(stain, (width // 2, height // 2), interpolation=cv2.INTER_AREA)  # blur at half-res
    base += cv2.resize(cv2.GaussianBlur(small, (0, 0), sigmaX=width * 0.015), (width, height))[..., None]
    base += np_rng.normal(0, rng.uniform(2.0, 4.0), (height, width, 1)).astype(
        np.float32
    )  # grain (luma, bcast)
    if tone is None:
        base[..., 0] *= rng.uniform(0.86, 0.96)  # drop blue -> warm/aged; lower end = browner foxed stock
    else:
        base *= np.array(tone.tint, np.float32)  # profile hue (warm/cool/yellowed/neutral)
    return np.clip(base, 0, 255).astype(np.uint8)


def content_mask(gray: np.ndarray, thresh: float = 12.0) -> np.ndarray:
    """Binary mask of structured content darker than the LOCAL paper: crisp ink AND faint verso
    bleed-through. Blurring the darker-than-local residual drops isolated fibre specks (kept as paper)
    while connected strokes / bleed-through survive — so mid-grey bleed no longer passes as blank."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, ker)  # local paper estimate (fills strokes + bleed)
    resid = np.clip(paper.astype(np.int16) - gray.astype(np.int16), 0, None).astype(np.float32)
    resid = cv2.GaussianBlur(resid, (0, 0), 3.0)  # high-freq fibre specks vanish; connected marks survive
    return (resid > thresh).astype(np.uint8)


def _window_candidates(gray: np.ndarray, ch: int, cw: int, grid: int) -> list[tuple[float, int, int]]:
    """Slide a ``(ch, cw)`` window over a coarse ``grid``; return ``(dark fraction, y, x)`` per window,
    emptiest first. Ranks by CRISP-ink darkness only — faint bleed alone doesn't disqualify a window."""
    h, w = gray.shape
    dark = (gray < 120).astype(np.uint8)
    ii = cv2.integral(dark)  # fast per-window dark fraction
    cands: list[tuple[float, int, int]] = []
    for y in range(0, max(1, h - ch), max(1, (h - ch) // grid)):
        for x in range(0, max(1, w - cw), max(1, (w - cw) // grid)):
            frac = float(ii[y + ch, x + cw] - ii[y, x + cw] - ii[y + ch, x] + ii[y, x]) / (ch * cw)
            cands.append((frac, y, x))
    cands.sort()
    return cands


def _inpainted_crop(img: np.ndarray, content: np.ndarray, y: int, x: int, ch: int, cw: int) -> np.ndarray:
    """The ``(ch, cw)`` crop at ``(y, x)`` with residual text INPAINTED out, brightened to a sane paper
    tone (texture kept) when the scan is dark — so the page + the global aging don't go near-black."""
    crop = img[y : y + ch, x : x + cw].copy()
    m = cv2.dilate(content[y : y + ch, x : x + cw], np.ones((5, 5), np.uint8), 2) * 255
    crop = cv2.inpaint(crop, m, 6, cv2.INPAINT_TELEA)  # paint residual text out -> blank paper
    med = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
    if med < 190:
        crop = np.clip(crop.astype(np.float32) * (190.0 / max(med, 1.0)), 0, 255).astype(np.uint8)
    return crop


def extract_paper_crops(
    scan_dir: str,
    out_dir: str,
    *,
    n_per_scan: int = 2,
    max_ink_frac: float = 0.04,
    grid: int = 8,
) -> int:
    """Save blank-paper background crops from each scan (real grime/aging, no text); return the count.

    A page-ish window is slid over a coarse ``grid`` and ranked by its text fraction (adaptive — darker
    than the LOCAL paper, so faded text counts). The ``n_per_scan`` emptiest *non-overlapping* windows
    below ``max_ink_frac`` are kept, and any residual text in them is INPAINTED out — so the crops carry
    the paper's real texture/aging/bleed-through but no handwriting (which would corrupt the labels).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scans = sorted(p for p in Path(scan_dir).iterdir() if p.suffix.lower() in _SCAN_EXT)
    total = 0
    for s in scans:
        img = cv2.imread(str(s))
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        ch = min(h - 2, max(int(h * 0.45), 1200))  # tall page-ish window, scaled to the scan
        cw = min(w - 2, max(int(ch * 0.6), 800))
        content = content_mask(g)  # crisp ink + faint verso bleed-through — inpainted out of the kept crop
        taken: list[tuple[int, int]] = []
        for frac, y, x in _window_candidates(g, ch, cw, grid):
            if frac > max_ink_frac:
                break
            if any(abs(y - ty) <= ch // 2 and abs(x - tx) <= cw // 2 for ty, tx in taken):
                continue  # overlaps a kept window
            cv2.imwrite(str(out / f"{s.stem}_{len(taken)}.png"), _inpainted_crop(img, content, y, x, ch, cw))
            taken.append((y, x))
            total += 1
            if len(taken) >= n_per_scan:
                break
    return total


def crops(
    scan_dir: Path = typer.Argument(..., help="folder of real document scans (.jpg/.png/.tif)"),
    out_dir: Path = typer.Argument(..., help="where to write the background crops"),
    per_scan: int = typer.Option(2, "--per-scan", help="crops to keep per scan"),
    max_ink: float = typer.Option(
        0.06, "--max-ink", help="reject windows with more than this dark-pixel fraction"
    ),
) -> None:
    """Extract blank-paper crops from real scans → a compose_page paper_dir."""
    n = extract_paper_crops(str(scan_dir), str(out_dir), n_per_scan=per_scan, max_ink_frac=max_ink)
    print(f"wrote {n} paper crops → {out_dir}")


def main() -> None:
    typer.run(crops)


if __name__ == "__main__":
    main()
