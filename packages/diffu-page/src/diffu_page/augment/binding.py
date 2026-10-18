"""Binding effects — the book spine's shadow, localized warp and focus falloff, plus the central
gutter shadow of a two-page opening."""

from __future__ import annotations

import random

import cv2
import numpy as np

from ..labels import LineLabel, Point
from ..layout import PageSpec
from .config import AugmentConfig
from .geometry import _remap_lines


def _spine_warp(
    page: np.ndarray, lines: list[LineLabel], rng: random.Random, spec: PageSpec, aug: AugmentConfig
) -> tuple[np.ndarray, list[LineLabel]]:
    """LOCALIZED binding distortion — only the strip at the spine warps; the rest of the page stays flat.

    Like a bound book pressed against a scanner: the paper can't lie flat at the gutter, so near the spine
    it compresses horizontally toward the binding and sags a little, decaying to zero within a randomized
    ``zone`` so it never shears the body text. Direction follows the page side (spine-left = recto,
    spine-right = verso). GT polygons ride the same displacement field.
    """
    H, W = page.shape[:2]
    zone = max(2.0, rng.uniform(0.12, 0.22) * W)  # how far the distortion reaches in; flat beyond it
    falloff = rng.uniform(1.7, 3.0)  # how fast it decays toward the body
    pull = rng.uniform(*aug.spine_warp) * zone  # horizontal compression toward the binding, px
    sag = rng.uniform(-0.014, 0.014) * H  # gentle vertical curl at the gutter (sign varies per page)

    x = np.arange(W, dtype=np.float32)
    dist = x if spec.binding_left else W - 1.0 - x  # distance from the spine edge
    t = np.clip(1.0 - dist / zone, 0.0, 1.0) ** falloff  # 1 at the spine -> 0 at the zone edge, flat beyond
    src_x = np.clip(x + (pull if spec.binding_left else -pull) * t, 0.0, W - 1)
    dy = sag * t  # per-column vertical shift, peaks at the spine

    mapx = np.broadcast_to(src_x, (H, W)).astype(np.float32)
    mapy = (np.arange(H, dtype=np.float32)[:, None] - dy[None, :]).astype(np.float32)
    warped = cv2.remap(page, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    fwd_x = np.interp(np.arange(W), src_x, np.arange(W, dtype=np.float32))  # invert the monotone x-map

    def ride(p: Point) -> Point:
        nx = fwd_x[min(max(int(p[0]), 0), W - 1)]
        return int(round(nx)), int(round(p[1] + sag * t[min(max(int(round(nx)), 0), W - 1)]))

    return warped, _remap_lines(lines, ride)


def _gutter_focus(page: np.ndarray, spec: PageSpec, rng: random.Random, aug: AugmentConfig) -> np.ndarray:
    """Soften focus toward the binding: the curved page near the spine drifts out of the scanner's focal
    plane, so blur ramps up at the spine edge and fades to sharp across the page. Label-safe (no pixels
    move)."""
    w = page.shape[1]
    reach = max(1.0, rng.uniform(0.05, 0.15) * w)  # how far the soft zone extends in from the spine
    dist = np.arange(w, dtype=np.float32)
    dist = dist if spec.binding_left else w - 1.0 - dist  # distance from the spine edge
    soft = (np.clip(1.0 - dist / reach, 0.0, 1.0) ** 1.5)[None, :, None]  # 1 at spine -> 0 across the page
    blurred = cv2.GaussianBlur(page, (0, 0), rng.uniform(*aug.gutter_blur)).astype(np.float32)
    return np.clip(page.astype(np.float32) * (1 - soft) + blurred * soft, 0, 255).astype(np.uint8)


def _binding_shadow(page: np.ndarray, spec: PageSpec, rng: random.Random, aug: AugmentConfig) -> None:
    """Darken the book spine like a real bound-volume scan: a narrow near-black **valley** right at the
    binding edge plus a broad soft **page-curvature fan** spreading inward, strongest at the top & bottom
    corners where the leaf curls. Strength, reach, corner-emphasis and falloff are all heavily randomized
    — from a barely-there tint to a deep gutter shadow. Confined to the text-free spine gutter that
    :func:`layout._page_elements` keeps empty — never over text.
    """
    h, w = page.shape[:2]
    x = np.arange(w, dtype=np.float32)  # distance in from the spine edge
    if not spec.binding_left:
        x = x[::-1]  # spine on the right -> distance in from the RIGHT edge
    valley = np.exp(-x / max(8.0, spec.binding_gutter * rng.uniform(0.30, 0.55)))  # dark seam at the spine
    curve = np.clip(1.0 - x / max(1.0, w * rng.uniform(0.09, 0.16)), 0.0, 1.0)  # fan reaches into the page,
    curve **= rng.uniform(1.2, 2.0)  # not just the thin gutter -> reads as a bound-volume page curl
    mid = rng.uniform(0.5, 0.78)  # corner emphasis: 1.0 at top/bottom curl, `mid` at the centre
    vfall = mid + (1.0 - mid) * np.linspace(-1.0, 1.0, h, dtype=np.float32) ** 2
    darken = rng.uniform(*aug.spine_shadow) * valley + rng.uniform(0.12, 0.34) * curve
    shadow = 1.0 - np.clip(darken[None, :] * vfall[:, None], 0.0, rng.uniform(0.6, 0.82))
    page[:] = np.clip(page.astype(np.float32) * shadow[..., None], 0.0, 255.0).astype(np.uint8)


def center_shadow(page: np.ndarray, rng: random.Random, aug: AugmentConfig) -> None:
    """Darken the CENTRAL gutter of a two-page opening — the valley where the bound leaves meet the platen.
    Confined to the text-free middle gutter the spread layout keeps empty, so it never falls on text."""
    w = page.shape[1]
    dist = np.abs(np.arange(w, dtype=np.float32) - w / 2.0)  # distance from the centre gutter
    valley = np.exp(-dist / max(8.0, w * 0.012 * rng.uniform(0.7, 1.4)))
    darken = np.clip(rng.uniform(*aug.spine_shadow) * valley, 0.0, rng.uniform(0.6, 0.8))
    page[:] = np.clip(page.astype(np.float32) * (1.0 - darken)[None, :, None], 0, 255).astype(np.uint8)
