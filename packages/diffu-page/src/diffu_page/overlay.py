"""Draw ground-truth labels onto a composed page so the IMAGE and the XML/labels can be verified together.

For every line it draws the polygon (green), the baseline (red), the word boxes (blue) and the reading-order
index (magenta) — so a glance at the overlay shows whether the labels actually fit the ink, sit on the right
line, and read in order. A debugging / audit visual, never part of the training output.
"""

from __future__ import annotations

import cv2
import numpy as np

from .labels import LineLabel

_POLYGON = (0, 200, 0)  # green
_BASELINE = (0, 0, 230)  # red
_WORD = (230, 120, 0)  # blue
_ORDER = (200, 0, 200)  # magenta


def _draw_line_overlay(out: np.ndarray, ln: LineLabel, *, words: bool) -> None:
    """Draw one line's polygon, baseline, word boxes and reading-order index onto ``out``."""
    if len(ln.polygon) >= 2:
        cv2.polylines(out, [np.array(ln.polygon, np.int32)], True, _POLYGON, 2, cv2.LINE_AA)
    if len(ln.baseline) >= 2:
        cv2.line(out, tuple(ln.baseline[0]), tuple(ln.baseline[-1]), _BASELINE, 2, cv2.LINE_AA)
    word_polys = [wd.polygon for wd in ln.words if len(wd.polygon) >= 2] if words else []
    for poly in word_polys:
        cv2.polylines(out, [np.array(poly, np.int32)], True, _WORD, 1, cv2.LINE_AA)
    if not ln.polygon:
        return
    x = min(p[0] for p in ln.polygon)
    y = min(p[1] for p in ln.polygon)
    cv2.putText(
        out,
        str(ln.order),
        (max(0, x - 34), y + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _ORDER,
        1,
        cv2.LINE_AA,
    )


def draw_label_overlay(page: np.ndarray, lines: list[LineLabel], *, words: bool = True) -> np.ndarray:
    """Return a copy of ``page`` with every line's polygon, baseline, word boxes and reading-order drawn on."""
    out = page.copy()
    for ln in lines:
        _draw_line_overlay(out, ln, words=words)
    return out
