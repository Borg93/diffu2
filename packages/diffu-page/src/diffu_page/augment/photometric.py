"""Photometric aging — lighting, noise, foxing, bleed-through, folds, microfilm and the JPEG pass.

No geometry moves here, so labels stay valid throughout.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .config import AugmentConfig


def _degrade(
    page: np.ndarray,
    grime: float,
    rng: random.Random,
    np_rng: np.random.Generator,
    aug: AugmentConfig,
    trace: dict[str, float] | None = None,
) -> np.ndarray:
    """Photometric aging on the flat page: one shared lighting field, an uneven-platen vignette, sensor
    noise, a warm tone shift, and (only on dirtier pages) brown foxing stains + verso bleed-through. No
    geometry, so labels stay valid. ``grime`` runs 0 (pristine) -> 1 (filthy) to span the real corpus."""
    H, W = page.shape[:2]
    out = page.astype(np.float32)
    gx = np.linspace(rng.uniform(0.90, 1.0), rng.uniform(0.95, 1.05), W, dtype=np.float32)
    gy = np.linspace(rng.uniform(0.92, 1.02), rng.uniform(0.90, 1.0), H, dtype=np.float32)
    out *= (gy[:, None] * gx[None, :])[..., None]  # one shared lighting field
    cy, cx = rng.uniform(0.38, 0.62), rng.uniform(0.38, 0.62)  # bright-centre -> dark-edge platen vignette
    xn = (
        np.arange(W, dtype=np.float32) / W - cx
    ) ** 2  # 1-D broadcast instead of a full H*W mgrid (3x faster)
    yn = (np.arange(H, dtype=np.float32) / H - cy) ** 2
    out *= (1.0 - rng.uniform(*aug.vignette) * (xn[None, :] + yn[:, None]))[..., None]
    # luma sensor noise as (H,W,1) broadcast over BGR — a third of the random draws of a full (H,W,3) field
    out += np_rng.normal(0, rng.uniform(1.5, 4.0) + 4.0 * grime, (H, W, 1)).astype(np.float32)
    out *= np.array([rng.uniform(0.82, 0.90), rng.uniform(0.95, 1.0), rng.uniform(1.02, 1.08)], np.float32)
    if grime > 0.12:
        stain = np.zeros((H, W), np.float32)
        for _ in range(rng.randint(1, int(7 * grime) + 2)):
            sx, sy = rng.randint(0, W), rng.randint(0, H)
            ax, ay = rng.randint(W // 30, W // 8), rng.randint(H // 40, H // 10)
            cv2.ellipse(stain, (sx, sy), (ax, ay), rng.randint(0, 180), 0, 360, rng.uniform(4, 12), -1)
        small = cv2.resize(
            stain, (W // 2, H // 2), interpolation=cv2.INTER_AREA
        )  # blur at half-res: ~4x cheaper
        foxing = cv2.resize(cv2.GaussianBlur(small, (0, 0), W * 0.01), (W, H))[..., None] * grime
        out -= foxing * np.array(aug.foxing_tint, np.float32)  # brown/sepia foxing (darkens B > G > R)
        gray = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ghost = cv2.GaussianBlur(cv2.flip(np.clip(225.0 - gray, 0, None), 1), (0, 0), 3)  # verso bleed
        bleed = rng.uniform(*aug.bleed)  # show-through varies a lot with paper thinness
        out -= ghost[..., None] * bleed
        if trace is not None:
            trace["foxing"] = float(grime)  # brown/sepia stain load -> a "water_stain"-class mark (C8)
            trace["bleed"] = float(bleed)  # verso show-through strength (C10)
    wash = rng.uniform(*aug.wash) * grime  # low-contrast brown damp wash — strongest on the dirtiest pages
    if wash > 0:
        out = 150.0 + (out - 150.0) * (1.0 - 0.6 * wash)  # pull toward mid-grey: faded, low-contrast
        out -= wash * 18.0 * np.array([1.3, 1.0, 0.6], np.float32)  # brownish darkening (B darkened most)
        if trace is not None:
            trace["wash"] = float(wash)  # damp brown wash (C8 water_stain-class)
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg_recompress(page: np.ndarray, grime: float, rng: random.Random) -> np.ndarray:
    """Re-encode so every line shares one compression signature; dirtier pages compress harder."""
    ok, enc = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, int(rng.randint(88, 96) - 25 * grime)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else page


def _folds(
    page: np.ndarray, rng: random.Random, aug: AugmentConfig, trace: dict[str, float] | None = None
) -> np.ndarray:
    """A fold/crease shadow line or a brown water tide-line. Photometric (multiply/subtract), label-safe."""
    if rng.random() >= aug.fold_prob:
        return page
    h, w = page.shape[:2]
    out = page.astype(np.float32)
    if rng.random() < 0.6:  # a soft crease shadow along a near-vertical or near-horizontal line
        vert = rng.random() < 0.5
        n = w if vert else h
        coord = np.arange(n, dtype=np.float32)
        band = np.exp(-((coord - rng.randint(0, n - 1)) ** 2) / (2 * rng.uniform(4, 12) ** 2))
        shade = 1.0 - rng.uniform(0.1, 0.3) * band
        out *= (shade[None, :] if vert else shade[:, None])[..., None]
        if trace is not None:
            trace["fold"] = 1.0  # a crease/fold shadow (C8)
    else:  # a brown water tide-line: an irregular brownish darkening across part of the page
        yy = np.arange(h, dtype=np.float32)[:, None]
        edge = rng.uniform(0.2, 0.8) * h
        tide = np.clip(1.0 - np.abs(yy - edge) / max(1.0, rng.uniform(0.05, 0.15) * h), 0.0, 1.0)
        out -= tide[..., None] * rng.uniform(8, 20) * np.array([1.4, 1.0, 0.5], np.float32)  # brown
        if trace is not None:
            trace["water_tide"] = 1.0  # brown water tide-line -> a "water_stain"-class mark (C8)
    return np.clip(out, 0, 255).astype(np.uint8)


def _microfilm(page: np.ndarray, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    """A microfilm-scan look — much Riksarkivet/GT-Lejonet material is filmed, not colour-flatbed: a
    high-contrast bitonal grey with film grain, horizontal density banding, vertical tramline scratches and
    a dark frame border. Photometric (the frame sits in the margin), so labels are untouched."""
    h, w = page.shape[:2]
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = (gray - 128.0) * rng.uniform(1.4, 2.2) + 128.0 * rng.uniform(0.9, 1.1)  # high-contrast bitonal-ish
    band = 1.0 + rng.uniform(0.02, 0.08) * np.sin(
        np.linspace(0.0, rng.uniform(2.0, 6.0) * np.pi, h, dtype=np.float32)
    )
    gray *= band[:, None]  # horizontal density banding
    gray += np_rng.normal(0, rng.uniform(4.0, 10.0), (h, w)).astype(np.float32)  # film grain
    for _ in range(rng.randint(0, 4)):  # vertical tramline scratches
        x = rng.randint(0, w - 1)
        gray[:, x] = gray[:, x] * 0.4 + rng.choice((0.0, 255.0)) * 0.6
    out = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bw = rng.randint(int(min(h, w) * 0.01), int(min(h, w) * 0.03) + 1)  # dark film frame border
    out[:bw, :] = out[-bw:, :] = out[:, :bw] = out[:, -bw:] = rng.randint(12, 30)
    return out
