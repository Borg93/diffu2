"""Render dispatch — turn a layout's boxes into ink rasters, per-line or batched.

Printed form-furniture (table headers, row numbers) is always typeset via the placeholder;
handwriting goes to the caller's renderer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import LineJob, LineRenderer, PlaceholderRenderer
from .diffu import BatchedDiffuRenderer

if TYPE_CHECKING:
    from ..layout import Layout, LineBox


def render_boxes(boxes: list[LineBox], renderer: LineRenderer) -> list[np.ndarray]:
    """Render each box to ink — the per-line path. Printed form-furniture (table headers, row numbers) is
    typeset; handwriting goes to ``renderer``. The batched equivalent is :func:`render_layouts`."""
    typeset = PlaceholderRenderer()
    return [
        typeset.render(box.text, box.h) if box.printed else renderer.render(box.text, box.h) for box in boxes
    ]


def render_layouts(
    renderer: BatchedDiffuRenderer, layouts: list[Layout], box_styles: list[list[str]]
) -> list[list[np.ndarray]]:
    """Batch-render every HANDWRITING line of every layout in one model pass; typeset (``printed``) boxes
    — form headers, row numbers — are rendered separately. ``box_styles[li][bi]`` is the writer-style path
    for that box, so a page can mix hands (e.g. a different style for marginalia). Returns inks grouped per
    layout (1:1 boxes)."""
    typeset = PlaceholderRenderer()
    jobs: list[LineJob] = []
    job_pos: list[tuple[int, int]] = []
    printed: dict[tuple[int, int], np.ndarray] = {}
    for li, (layout, styles) in enumerate(zip(layouts, box_styles, strict=True)):
        for bi, box in enumerate(layout.boxes):
            if box.printed:
                printed[(li, bi)] = typeset.render(box.text, box.h)
            else:
                jobs.append(LineJob(text=box.text, height=box.h, style_path=styles[bi]))
                job_pos.append((li, bi))
    model_inks = renderer.render_jobs(jobs)
    by_pos = dict(zip(job_pos, model_inks, strict=True))
    return [
        [printed[(li, bi)] if (li, bi) in printed else by_pos[(li, bi)] for bi in range(len(layout.boxes))]
        for li, layout in enumerate(layouts)
    ]
