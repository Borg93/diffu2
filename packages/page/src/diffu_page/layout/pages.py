"""A scan's pages. Most archive scans are openings, two facing pages in one image: :func:`scan_pages` splits one at
its gutter (:func:`gutter`) into its pages, each in its own coordinates, so a page is drawn and labelled as a single
page, never as a spread. Any other scan is one page.

The gutter is read from the lines alone. On the GT (27 Oct) it cuts the 27,169 openings it finds at 0.38-0.62 of
their width in 98.6 % of them, and no table in two; where the transcribers' lines or a table run over the fold, the
only free bands lie inside a page and the cut lands there (about 1 %): telling them apart needs the scan itself."""

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
    text line's box crosses and that parts no table, rather one that parts no text region either. A register's columns
    leave bands inside a page wider than its fold, and a transcriber's region may run over the fold. None for a single
    page: a scan no wider than tall or with no line, or one with no such band (its lines or a table across its
    middle)."""
    if layout.width <= layout.height or not layout.lines:
        return None
    widest_first = sorted(_free_bands(layout), key=lambda band: band[0] - band[1])
    middles = [(left + right) / 2 for left, right in widest_first]
    tables_whole = [x for x in middles if not _parts(layout, x, tables=True)]
    regions_whole = [x for x in tables_whole if not _parts(layout, x, tables=False)]
    candidates = regions_whole or tables_whole
    return candidates[0] if candidates else None


def _free_bands(layout: Layout) -> list[tuple[float, float]]:
    """The bands of the scan's middle (:data:`MIDDLE`) that no text line's box crosses, left to right."""
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
    return free


def _parts(layout: Layout, x: float, *, tables: bool) -> bool:
    """Whether a cut at ``x`` leaves lines of one table (``tables``), or of one table or text region, on both sides."""
    sides: dict[str, set[bool]] = {}
    for line in layout.lines:
        if line.cell is not None:
            group = f"table:{line.cell.table}"
        elif tables:
            continue
        else:
            group = f"region:{line.block}"
        sides.setdefault(group, set()).add(line.middle_x < x)
    return any(len(seen) == 2 for seen in sides.values())


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
