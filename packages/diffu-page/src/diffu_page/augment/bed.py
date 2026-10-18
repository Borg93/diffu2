"""Scanner bed — mount the sheet on a dark bed with real archival edge contours and page-holder
hardware. :class:`PaperBounds` records where the leaf landed so margin damage stays on the paper."""

from __future__ import annotations

import random

import cv2
import numpy as np
from pydantic import BaseModel

from ..labels import LineLabel, Point
from ..layout import PageSpec
from .config import AugmentConfig
from .geometry import _remap_lines


def _edge_profile(n: int, depth: float, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    """How far edge damage eats in along an edge of length ``n`` (pixels, >= 0): smooth waviness + fine
    fray + the occasional shallow nick — a real archival fore-edge, not a straight cut."""
    cp = np_rng.uniform(0.15, 1.0, rng.randint(5, 12))  # control points -> gentle low-frequency waviness
    prof = np.interp(np.linspace(0, len(cp) - 1, n), np.arange(len(cp)), cp) * depth
    prof += np_rng.uniform(0.0, depth * 0.15, n)  # fine fray (subtle)
    for _ in range(rng.randint(0, 2)):  # rare, shallow nick — not a deep tear
        c, half = rng.randint(0, n - 1), rng.randint(max(2, n // 90), max(3, n // 35))
        lo, hi = max(0, c - half), min(n, c + half)
        prof[lo:hi] += rng.uniform(0.4 * depth, 1.4 * depth) * np.hanning(hi - lo)
    return prof.astype(np.float32)


def _draw_clasps(page: np.ndarray, rng: random.Random, aug: AugmentConfig) -> None:
    """An occasional dark metal page-holder clasp intruding from a scanner-bed edge, usually near a CORNER,
    as in archive scans. Drawn as a SOFT-edged dark bar (a blurred mask darkened through) — not a hard
    rectangle — so it reads as metal/shadow rather than a UI box. Over the bed/outer margin."""
    if rng.random() >= aug.clasp_prob:
        return
    h, w = page.shape[:2]
    m = min(h, w)
    reach = rng.randint(int(m * 0.035), int(m * 0.08))  # how far it protrudes in from the edge
    wide = rng.randint(int(m * 0.02), int(m * 0.04))
    edge = rng.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        corner = (rng.randint(0, max(1, int(w * 0.2))), rng.randint(int(w * 0.8), max(2, w - wide)))
        x0 = rng.choice(corner)  # bias toward a corner, not mid-edge
        x1, (y0, y1) = x0 + wide, ((0, reach) if edge == "top" else (h - reach, h))
    else:
        corner = (rng.randint(0, max(1, int(h * 0.2))), rng.randint(int(h * 0.8), max(2, h - wide)))
        y0 = rng.choice(corner)
        y1, (x0, x1) = y0 + wide, ((0, reach) if edge == "left" else (w - reach, w))
    mask = np.zeros((h, w), np.float32)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), m * 0.005)  # soft, rounded, fuzzy edges — metal, not a UI box
    page[:] = np.clip(
        page.astype(np.float32) * (1.0 - mask * rng.uniform(0.55, 0.85))[..., None], 0, 255
    ).astype(np.uint8)


class PaperBounds(BaseModel):
    """The paper leaf's content rect within the (bed-mounted) canvas. Margin damage that must land ON the
    sheet — punch/worm holes — samples its bands from this rect, not the full canvas."""

    model_config = {"frozen": True}

    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def full(cls, page: np.ndarray) -> PaperBounds:
        h, w = page.shape[:2]
        return cls(x0=0, y0=0, x1=w, y1=h)


def _mount_on_bed(
    page: np.ndarray,
    lines: list[LineLabel],
    rng: random.Random,
    np_rng: np.random.Generator,
    spec: PageSpec,
    aug: AugmentConfig,
) -> tuple[np.ndarray, list[LineLabel], PaperBounds]:
    """Drop the sheet onto a dark scanner bed: irregular dark borders, grey sensor noise and dark corners,
    and give every FREE edge (all but the bound spine) a real archival contour — wavy, frayed, occasionally
    nicked, darkened/worn toward the edge — instead of a clean rectangle. The document is resized into the
    inner rect; its polygons scale + offset to track it, and the damage stays in the margin clear of text.
    """
    H, W = page.shape[:2]
    insets = [rng.uniform(0.0, 0.05) for _ in range(4)]  # top, bottom, left, right
    insets[rng.randint(0, 3)] += rng.uniform(0.0, 0.04)  # one edge usually shows more bed (fore-edge)
    if spec.binding:  # the bound spine runs into the gutter — no bed gap / hole on that edge
        insets[2 if spec.binding_left else 3] = 0.0
    pt, pb = int(H * insets[0]), int(H * insets[1])
    pl, pr = int(W * insets[2]), int(W * insets[3])
    ih, iw = H - pt - pb, W - pl - pr

    bed = np.full((H, W, 3), rng.uniform(10, 40), np.float32)
    bed += np_rng.normal(0, rng.uniform(5, 16), (H, W, 3)).astype(np.float32)  # grey sensor noise
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bed -= (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * rng.uniform(20, 70))[..., None]  # dark corners

    free = {"left": True, "right": True, "top": True, "bottom": True}
    if spec.binding:
        free["left" if spec.binding_left else "right"] = False  # the bound spine edge stays clean
    depth = rng.uniform(*aug.edge_damage) * min(H, W)  # subtle edge irregularity, not a rough cut
    blank_h, blank_w = np.zeros(ih, np.float32), np.zeros(iw, np.float32)
    left = _edge_profile(ih, depth, rng, np_rng) if free["left"] else blank_h
    right = _edge_profile(ih, depth, rng, np_rng) if free["right"] else blank_h
    top = _edge_profile(iw, depth, rng, np_rng) if free["top"] else blank_w
    bottom = _edge_profile(iw, depth, rng, np_rng) if free["bottom"] else blank_w
    left_b, right_b = np.full(H, pl, np.float32), np.full(H, pl + iw, np.float32)
    top_b, bot_b = np.full(W, pt, np.float32), np.full(W, pt + ih, np.float32)
    left_b[pt : pt + ih], right_b[pt : pt + ih] = pl + left, pl + iw - right
    top_b[pl : pl + iw], bot_b[pl : pl + iw] = pt + top, pt + ih - bottom
    inside = (
        (xx >= left_b[:, None]) & (xx <= right_b[:, None]) & (yy >= top_b[None, :]) & (yy <= bot_b[None, :])
    )

    doc = bed.copy()
    doc[pt : pt + ih, pl : pl + iw] = cv2.resize(page, (iw, ih), interpolation=cv2.INTER_AREA)
    dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
    wear = 0.72 + 0.28 * np.clip(dist / max(4.0, rng.uniform(0.006, 0.016) * min(H, W)), 0.0, 1.0)
    doc *= wear[..., None]  # paper darkens / wears GENTLY toward the irregular edge
    mask = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 1.6)[..., None]  # soft-feather the boundary
    out = np.clip(bed * (1 - mask) + doc * mask, 0, 255).astype(np.uint8)
    _draw_clasps(out, rng, aug)  # page-holder hardware intruding from the bed edges

    sx, sy = iw / W, ih / H

    def place(p: Point) -> Point:
        return int(p[0] * sx) + pl, int(p[1] * sy) + pt

    return out, _remap_lines(lines, place), PaperBounds(x0=pl, y0=pt, x1=pl + iw, y1=pt + ih)
