"""Tests for the GT overlay visual — it returns a same-shape copy with the labels drawn on."""

from __future__ import annotations

import numpy as np
from diffu_page.labels import LineLabel, WordLabel
from diffu_page.overlay import draw_label_overlay

_LINES = [
    LineLabel(
        text="hej va",
        polygon=[(10, 20), (90, 20), (90, 44), (10, 44)],
        baseline=[(10, 40), (90, 40)],
        words=[
            WordLabel(text="hej", polygon=[(10, 20), (45, 20), (45, 44), (10, 44)]),
            WordLabel(text="va", polygon=[(55, 20), (90, 20), (90, 44), (55, 44)]),
        ],
        order=0,
    )
]


def test_overlay_returns_a_page_shaped_copy_with_labels_drawn():
    page = np.full((120, 200, 3), 255, np.uint8)
    before = page.copy()

    out = draw_label_overlay(page, _LINES)

    assert out.shape == page.shape and out.dtype == page.dtype
    assert not np.array_equal(out, page), "polygon/baseline/word boxes must actually be drawn"
    assert np.array_equal(page, before), "the input page must not be mutated (overlay is a copy)"


def test_overlay_of_no_lines_is_the_unmarked_page():
    page = np.full((60, 80, 3), 255, np.uint8)
    out = draw_label_overlay(page, [])
    assert out.shape == page.shape
    assert np.array_equal(out, page)
