"""Whole-sheet geometry — warps and rotations whose displacement the GT polygons ride.

The shared :func:`_remap_lines` point-map helper lives here, alongside the global perspective
wobble, the whole-page rotation, and the final label clamp.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import cv2
import numpy as np

from ..labels import LineLabel, Point
from ..layout import PageSpec
from .config import AugmentConfig


def _remap_lines(lines: list[LineLabel], fn: Callable[[Point], Point]) -> list[LineLabel]:
    """Apply a point map ``fn`` to every line + word polygon, returning fresh labels."""
    return [
        ln.model_copy(
            update={
                "polygon": [fn(p) for p in ln.polygon],
                "baseline": [fn(p) for p in ln.baseline],
                "words": [w.model_copy(update={"polygon": [fn(p) for p in w.polygon]}) for w in ln.words],
            }
        )
        for ln in lines
    ]


def _scanner_warp(
    page: np.ndarray, lines: list[LineLabel], rng: random.Random, aug: AugmentConfig
) -> tuple[np.ndarray, list[LineLabel]]:
    """A small GLOBAL perspective wobble — the whole sheet read as one slightly-skewed photograph."""
    H, W = page.shape[:2]
    jx, jy = W * rng.uniform(0.0, aug.skew), H * rng.uniform(0.0, aug.skew)
    corners = [(0.0, 0.0), (float(W), 0.0), (float(W), float(H)), (0.0, float(H))]
    src = np.float32(corners)
    dst = np.float32([[x + rng.uniform(-jx, jx), y + rng.uniform(-jy, jy)] for x, y in corners])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, M, (W, H), borderValue=tuple(int(v) for v in page[0, 0]))

    def project(p: Point) -> Point:
        d = M @ np.array([p[0], p[1], 1.0])
        return int(d[0] / d[2]), int(d[1] / d[2])

    return warped, _remap_lines(lines, project)


def _page_rotate(
    page: np.ndarray, lines: list[LineLabel], rng: random.Random, aug: AugmentConfig
) -> tuple[np.ndarray, list[LineLabel]]:
    """Rotate the WHOLE sheet a few degrees on the bed (sign random) — real scans are routinely a few
    degrees off-square, and the baseline/layout stage must tolerate it. GT rides the rotation."""
    deg = rng.uniform(*aug.rotate) * rng.choice((-1.0, 1.0))
    if abs(deg) < 1e-3:
        return page, lines
    h, w = page.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    border = tuple(int(v) for v in page[0, 0])
    warped = cv2.warpAffine(page, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)

    def rotate(p: Point) -> Point:
        return (
            int(round(M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2])),
            int(round(M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2])),
        )

    return warped, _remap_lines(lines, rotate)


def _clamp_lines(lines: list[LineLabel], spec: PageSpec, w: int, h: int) -> list[LineLabel]:
    """Keep every label on the sheet and off the reserved spine gutter — no overflow, no text on the
    binding — after the warps have moved the polygons."""
    x_lo = (spec.margin + spec.binding_gutter) if (spec.binding and spec.binding_left) else 0
    x_hi = (w - spec.margin - spec.binding_gutter) if (spec.binding and not spec.binding_left) else w - 1

    def clamp(p: Point) -> Point:
        return min(max(p[0], x_lo), x_hi), min(max(p[1], 0), h - 1)

    return _remap_lines(lines, clamp)
