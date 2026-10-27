"""Ruled lines found on a scan (diffu_page.marking.rules): a form's rules found where they run among its writing, and
handwriting strokes, a page's edge and an opening's fold not taken for rules."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import pytest

from diffu_page.layout import Layout, PageBox, TextLine
from diffu_page.marking.rules import find_rules

type Point = tuple[int, int]
type Segment = tuple[Point, Point]

PAPER, INK, BACKGROUND = 225, 40, 30
COLUMNS = ((140, 760), (840, 1460))
ROWS = tuple(200 + 80 * k for k in range(10))  # each row's top; a text line box is 40 px high
LINES = [(x0, top, x1, top + 40) for top in ROWS for x0, x1 in COLUMNS]  # two columns of ten lines, 1600 x 1200 px
RULED = [  # the form's rules: across its top and bottom, down its sides and between its columns
    ((100, 160), (1500, 160)),
    ((100, 1000), (1500, 1000)),
    ((100, 160), (100, 1000)),
    ((800, 160), (800, 1000)),
    ((1500, 160), (1500, 1000)),
]


def _scan(
    size: tuple[int, int], boxes: Sequence[PageBox], *, lines: Sequence[Segment], dark: Sequence[PageBox]
) -> np.ndarray:
    """A grey scan of ``size`` (width, height): paper, the ``dark`` background beyond a page, a few words of scribbled
    handwriting in each text line box, and straight 2 px ``lines``."""
    rng = np.random.default_rng(0)
    scan = np.full((size[1], size[0]), PAPER, np.uint8)
    for x0, y0, x1, y1 in dark:
        cv2.rectangle(scan, (round(x0), round(y0)), (round(x1), round(y1)), BACKGROUND, -1)
    for x0, y0, x1, y1 in boxes:
        for word in range(round(x0) + 5, round(x1) - 35, 45):
            points = np.stack([rng.integers(word, word + 35, 8), rng.integers(round(y0) + 6, round(y1) - 6, 8)], axis=1)
            cv2.polylines(scan, [points.astype(np.int32)], isClosed=False, color=INK, thickness=2)
    for start, end in lines:
        cv2.line(scan, start, end, INK, 2)
    return scan


def _layout(size: tuple[int, int], boxes: Sequence[PageBox]) -> Layout:
    lines = [TextLine(text="Anno 1720", box=box, baseline_y=box[3] - 10) for box in boxes]
    return Layout(width=size[0], height=size[1], lines=lines)


def test_find_rules_finds_a_forms_ruled_lines_where_they_run_among_its_writing() -> None:
    size = (1600, 1200)
    scan = _scan(size, LINES, lines=RULED, dark=[])

    rules = find_rules(scan, _layout(size, LINES))

    assert [(rule.x0, rule.y0, rule.x1, rule.y1) for rule in rules] == [
        pytest.approx((*start, *end), abs=3) for start, end in RULED
    ]


SHORT_LINE = [(140, 280, 400, 320)]  # the second line of the left column, ending early
OPENING = [(x0, top, x1, top + 40) for top in ROWS for x0, x1 in ((150, 950), (1050, 1850))]  # 2000 x 1200 px


@pytest.mark.parametrize(
    ("size", "boxes", "lines", "dark"),
    [
        pytest.param((1600, 1200), LINES, [((200, 220), (500, 220))], [], id="a-dash-in-a-text-line"),
        pytest.param(
            (1600, 1200),
            [box for box in LINES if box[:2] != (140, 280)] + SHORT_LINE,
            [((402, 300), (740, 300))],
            [],
            id="a-filler-running-on-from-a-text-line",
        ),
        pytest.param((1600, 1200), LINES, [((200, 260), (400, 260))], [], id="a-stroke-five-lines-long"),
        pytest.param((1600, 1200), LINES, [((0, 150), (1600, 150))], [(0, 0, 1600, 147)], id="a-pages-top-edge"),
        pytest.param((2000, 1200), OPENING, [((1000, 200), (1000, 960))], [], id="an-openings-fold"),
    ],
)
def test_find_rules_takes_no_stroke_edge_or_fold_for_a_rule(
    size: tuple[int, int], boxes: Sequence[PageBox], lines: Sequence[Segment], dark: Sequence[PageBox]
) -> None:
    scan = _scan(size, boxes, lines=lines, dark=dark)

    assert find_rules(scan, _layout(size, boxes)) == []
