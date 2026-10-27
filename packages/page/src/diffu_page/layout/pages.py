"""A scan's pages. Most archive scans are openings, two facing pages in one image: :func:`scan_pages` splits one at
its gutter (:func:`gutter`) into its pages, each in its own coordinates, so a page is drawn and labelled as a single
page, never as a spread. Any other scan is one page."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from diffu_page.layout.model import Layout, PageBox, Rule, TextLine

MIDDLE = (0.3, 0.7)  # where an opening's gutter is looked for, as fractions of the scan's width


class PageSide(StrEnum):
    """A page of a two-page opening: its left or its right one."""

    LEFT = "left"
    RIGHT = "right"


class ScanPage(BaseModel):
    """One page of a scan: its side of an opening (None for a single page), its rectangle on the scan (px) and its
    layout in its own coordinates."""

    model_config = {"frozen": True}

    side: PageSide | None
    rect: PageBox
    layout: Layout


def gutter(layout: Layout) -> float | None:
    """An opening's gutter, x in page px: the middle of the widest band of the scan's middle (:data:`MIDDLE`) that no
    text line's box crosses. None for a single page: a scan no wider than tall or with no line, or one whose middle
    its lines cross throughout."""
    if layout.width <= layout.height or not layout.lines:
        return None
    low, high = (fraction * layout.width for fraction in MIDDLE)
    free: list[tuple[float, float]] = []
    reach = low  # how far right the lines swept so far reach, at least low
    for x0, x1 in sorted((line.box[0], line.box[2]) for line in layout.lines):
        if x0 > reach:
            free.append((reach, min(x0, high)))
        reach = max(reach, x1)
        if reach >= high:
            break
    if reach < high:
        free.append((reach, high))
    if not free:
        return None
    left, right = max(free, key=lambda band: band[1] - band[0])
    return (left + right) / 2


def scan_pages(layout: Layout) -> list[ScanPage]:
    """The scan's pages: an opening's two pages, split at its gutter (a line on the side its middle is on), those that
    hold a line, each in its own coordinates; any other scan whole, as its one page."""
    split = gutter(layout)
    if split is None:
        return [ScanPage(side=None, rect=(0.0, 0.0, float(layout.width), float(layout.height)), layout=layout)]
    edge = round(split)
    sides = (
        _page(layout, PageSide.LEFT, (0, edge), [line for line in layout.lines if line.middle_x < edge]),
        _page(layout, PageSide.RIGHT, (edge, layout.width), [line for line in layout.lines if line.middle_x >= edge]),
    )
    return [page for page in sides if page.layout.lines]


def _page(layout: Layout, side: PageSide, span: tuple[int, int], lines: list[TextLine]) -> ScanPage:
    """The page of the scan between the x of ``span``, holding ``lines``, in its own coordinates: its lines moved by
    its left edge, the scan's ruled lines cut to it and moved with them."""
    left, right = span
    moved = [
        line.model_copy(update={"box": (line.box[0] - left, line.box[1], line.box[2] - left, line.box[3])})
        for line in lines
    ]
    rules = [
        Rule(x0=max(rule.x0, left) - left, y0=rule.y0, x1=min(rule.x1, right) - left, y1=rule.y1)
        for rule in layout.rules
        if rule.x0 < right and rule.x1 > left
    ]
    return ScanPage(
        side=side,
        rect=(float(left), 0.0, float(right), float(layout.height)),
        layout=Layout(width=right - left, height=layout.height, lines=moved, rules=rules),
    )
