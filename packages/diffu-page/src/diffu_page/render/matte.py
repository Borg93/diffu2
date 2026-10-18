"""Ink matting — turn a generated line image into a BGRA ink-on-transparent raster.

The compositor never re-renders the ink — it alpha-OVERs the model's *real pixels*; the matte here
is what separates real strokes from the model's grey writing-region wash and a real crop's white
padding, so only ink survives onto the page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


def ink_from_pil(pil: Image, height: int) -> np.ndarray:
    """A generated line image -> BGRA, resized to ``height`` (aspect kept).

    The compositor never re-renders the ink — it alpha-OVERs the model's *real pixels*. So we carry:

    - **BGR** = the model's actual line patch, verbatim (ink + its paper), resized.
    - **alpha** (A) = a soft ink matte (see :func:`ink_matte`) that keeps the strokes and maps the line's
      own grey writing-region wash — and any white padding around a real crop — to 0, so only real ink
      survives and sits directly on the page paper (no off-paper box or ``[ ]`` frame around the line).
    """
    bgr = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR).astype(np.float32)
    gray = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    alpha = ink_matte(gray)
    h, w = gray.shape
    if h != height:
        nw = max(1, int(w * height / h))
        bgr = cv2.resize(bgr, (nw, height), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (nw, height), interpolation=cv2.INTER_AREA)
    bgra = np.empty((*alpha.shape, 4), np.uint8)
    bgra[..., :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    bgra[..., 3] = (alpha * 255.0).astype(np.uint8)
    return bgra


def ink_matte(gray: np.ndarray) -> np.ndarray:
    """Soft ink coverage in ``[0, 1]`` via LOCAL background subtraction — keeps the full strokes the model
    drew while flattening its grey writing-region wash to 0.

    A flat brightness threshold can't separate the lighter parts of a stroke from the grey wash (both are
    mid-grey), so it either keeps the wash as a box OR eats the strokes down to broken fragments. Instead,
    estimate the LOCAL background and take ``bg - gray``: a pixel darker than its surroundings (the whole
    stroke, core + lighter edges) scores high, while the flat uniform wash (== its own local background)
    scores ~0. A small floor removes the faint wash residual.

    Two details keep this clean on *real* line crops (grey writing-region on white padding):

    - **Median, not morphological close, for the background.** Grayscale close dilates first, so at a step
      edge (the grey wash meeting the white padding at a line's start/end) it drags the bright padding
      *into* the wash, and ``bg - gray`` lights up the whole region perimeter -> an off-paper ``[ ]`` frame
      around every seated line. A median is robust to that step, so no frame. It also collapses solid dark
      BANDS (a scan smudge reads as its own background -> ~0 ink) that close would keep.
    - **Flatten the white padding to the wash first.** With the padding still pure white, even the median
      window straddling the wash/padding boundary leaves a faint tick; mapping it to the local wash removes
      the step entirely. On model output (no white padding) this is a no-op."""
    g = gray.copy()
    body = gray[gray < 250.0]  # everything but the crop's bright white padding
    if body.size:
        g[gray >= 250.0] = float(
            np.percentile(body, 60)
        )  # flatten padding to the local wash: kill the step edge
    k = max(11, (g.shape[0] // 3) | 1)  # window a bit wider than a stroke, to see across it to the background
    bg = cv2.medianBlur(g.astype(np.uint8), k).astype(
        np.float32
    )  # median bg: robust to step edges + solid bands
    ink = bg - g  # how much darker than the LOCAL background -> the full stroke
    positive = ink[ink > 2.0]
    scale = float(np.percentile(positive, 90)) if positive.size else 30.0
    alpha = np.clip(ink / max(scale, 10.0), 0.0, 1.0)
    alpha = np.clip((alpha - 0.22) / 0.78, 0.0, 1.0)  # floor: drop the faint wash residual, keep the strokes
    return _suppress_blobs(alpha)


def _suppress_blobs(alpha: np.ndarray) -> np.ndarray:
    """Zero out solid high-fill components — source-scan smudges and the model's grey wash BANDS — while
    leaving handwriting untouched. A band is a filled block (large area, high bbox fill, tall); a stroke is
    a thin curve (low fill) or small, so the thresholds never catch real writing (verified on real crops)."""
    h = alpha.shape[0]
    mask = (alpha > 0.30).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = alpha.copy()
    for c in range(1, count):
        area = int(stats[c, cv2.CC_STAT_AREA])
        cw, ch = int(stats[c, cv2.CC_STAT_WIDTH]), int(stats[c, cv2.CC_STAT_HEIGHT])
        if area > 0.03 * alpha.size and area / max(1, cw * ch) > 0.5 and ch > 0.35 * h:
            out[labels == c] = 0.0
    return out
