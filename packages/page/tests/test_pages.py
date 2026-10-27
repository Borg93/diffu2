"""A scan's pages: a two-page opening split at its gutter, the widest band of its middle that no line crosses and that
parts no table (rather no text region either), into single pages, each in its own coordinates; any other scan is one
page."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffu_page.layout import (
    CellRef,
    Layout,
    LineKind,
    PageSide,
    Rule,
    TextLine,
    gutter,
    load_xml_layout,
    scan_pages,
)


def _line(x0: float, x1: float, region: str = "") -> TextLine:
    """A line from ``x0`` to ``x1``, near the top of its page, in the text region ``region``."""
    return TextLine(text="den fjerde dagen i maj", box=(x0, 200.0, x1, 260.0), baseline_y=250.0, block=region)


def _cell(x0: float, x1: float, table: str, column: int) -> TextLine:
    """A table cell's line from ``x0`` to ``x1``: column ``column`` of the first row of ``table``."""
    return TextLine(
        text="Anders Persson",
        box=(x0, 200.0, x1, 260.0),
        baseline_y=250.0,
        kind=LineKind.TABLE,
        block=f"{table}_c{column}",
        cell=CellRef(table=table, row=0, column=column),
    )


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(Layout(width=1000, height=1400, lines=[_line(100, 400), _line(600, 900)]), id="portrait"),
        pytest.param(Layout(width=2000, height=1000), id="no-line"),
    ],
)
def test_gutter_is_none_for_a_single_page(layout: Layout) -> None:
    assert gutter(layout) is None


def test_gutter_of_a_register_opening_is_its_fold_not_the_gap_between_a_table_s_columns() -> None:
    # a register of names and numbers on each page: the left page's gap between its columns (600-850 of the scan's
    # middle) is wider than the fold (950-1100), but a gutter never parts a table
    columns = [(100, 500, "left", 0), (850, 950, "left", 1), (1100, 1500, "right", 0), (1850, 1950, "right", 1)]
    layout = Layout(width=2000, height=1200, lines=[_cell(*column) for column in columns])

    assert gutter(layout) == 1025.0


def test_a_scan_with_one_table_across_its_middle_is_one_page() -> None:
    columns = [(100, 500, "ledger", 0), (850, 950, "ledger", 1), (1100, 1500, "ledger", 2), (1850, 1950, "ledger", 3)]
    layout = Layout(width=2000, height=1200, lines=[_cell(*column) for column in columns])

    assert [(page.side, page.rect) for page in scan_pages(layout)] == [(None, (0.0, 0.0, 2000.0, 1200.0))]


@pytest.mark.parametrize(
    ("regions", "expected"),
    [
        pytest.param(("left", "left", "right"), 1025.0, id="a band that parts no region over a wider one"),
        pytest.param(("both", "both", "both"), 725.0, id="the widest band when each parts a region"),
    ],
)
def test_gutter_parts_no_text_region_where_a_band_allows(regions: tuple[str, str, str], expected: float) -> None:
    # bands of the scan's middle: 600-850 and 950-1100; a transcriber's region may run over the fold
    spans = [(100, 500), (850, 950), (1100, 1900)]
    layout = Layout(
        width=2000, height=1200, lines=[_line(*span, region) for span, region in zip(spans, regions, strict=True)]
    )

    assert gutter(layout) == expected


def test_scan_pages_split_an_opening_at_its_gutter_into_its_two_pages_each_in_its_own_coordinates(
    opening_page: Path,
) -> None:
    layout = load_xml_layout(opening_page)

    pages = scan_pages(layout)

    # the gutter: the middle of 1150-1400, the wider of the bands of the scan's middle either side of the note
    assert [(page.side, page.rect, page.layout.width, page.layout.height) for page in pages] == [
        (PageSide.LEFT, (0.0, 0.0, 1275.0, 1200.0), 1275, 1200),
        (PageSide.RIGHT, (1275.0, 0.0, 2500.0, 1200.0), 1225, 1200),
    ]
    assert [[(line.text, line.box, line.baseline_y) for line in page.layout.lines] for page in pages] == [
        [(f"left line {i}", (100.0, 240.0 + 100 * i, 1000.0, 320.0 + 100 * i), 300.0 + 100 * i) for i in range(4)]
        + [("No 12", (1050.0, 340.0, 1150.0, 420.0), 400.0)],  # the inner margin's note, on the left page
        [(f"right line {i}", (125.0, 240.0 + 100 * i, 1125.0, 320.0 + 100 * i), 300.0 + 100 * i) for i in range(3)],
    ]
    assert [page.layout.rules for page in pages] == [
        [Rule(x0=100.0, y0=330.0, x1=1275.0, y1=332.0)],
        [Rule(x0=0.0, y0=330.0, x1=1125.0, y1=332.0)],
    ]


def test_scan_pages_of_an_opening_leave_out_a_page_with_no_line() -> None:
    layout = Layout(width=2000, height=1000, lines=[_line(100, 700)])  # the gutter at 1050, the right page blank

    pages = scan_pages(layout)

    assert [(page.side, page.rect) for page in pages] == [(PageSide.LEFT, (0.0, 0.0, 1050.0, 1000.0))]
