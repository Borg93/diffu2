"""Render stage — turn one line's text into an ink-on-transparent raster for compositing.

:class:`LineRenderer` is the seam between "what to write" and "how it looks". Two implementations:

- :class:`PlaceholderRenderer` draws gray text with OpenCV (no model, no GPU, no torch) so the
  layout + compositor are runnable and testable *today*.
- :class:`DiffuRenderer` wraps :func:`diffu.generate.generate_line` (lazy torch import) — one writer
  style per page, enforced by taking the style reference at construction.

Output convention (both): a ``BGRA`` ``uint8`` array whose **alpha channel is the ink coverage**
(255 = full ink, 0 = paper). BGR is left neutral — the ink *colour* is chosen later by the
compositor from the local paper, so blending stays adaptive.

Split into cohesive submodules — ``base`` (the seam + placeholder), ``matte`` (ink matting),
``diffu`` (the Diffu renderers), ``eruku`` (the Eruku renderer) and ``dispatch``
(:func:`render_boxes` / :func:`render_layouts`) — this package re-exports the old flat-module
surface.
"""

from __future__ import annotations

from .base import LineJob, LineRenderer, PlaceholderRenderer
from .diffu import BatchedDiffuRenderer, DiffuRenderer
from .dispatch import render_boxes, render_layouts
from .eruku import ErukuRenderer
from .matte import ink_from_pil, ink_matte

__all__ = [
    "BatchedDiffuRenderer",
    "DiffuRenderer",
    "ErukuRenderer",
    "LineJob",
    "LineRenderer",
    "PlaceholderRenderer",
    "ink_from_pil",
    "ink_matte",
    "render_boxes",
    "render_layouts",
]
