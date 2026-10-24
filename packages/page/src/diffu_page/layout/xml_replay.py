"""A page layout from a real PAGE or ALTO XML file.

This is the one reader: the training-crop builder (:mod:`diffu_page.data.build_crops`) cuts its crops from this
layout and the planner plans generation from it, so a page is planned exactly as its training crops were cut: the same
outlines, baselines, region roles and Transkribus reading order. An element without an id is keyed by its place on the
page, so it stays a region of its own. A PAGE table is read in either form a template comes in: Transkribus' ``TableCell``
(``@row``, ``@col``) or PAGE 2019's nested ``TextRegion`` with a ``TableCellRole``. A line's ``@production`` (else its
region's, its cell's or its table's) says whether it is printed; ``SeparatorRegion``s are the page's ruled lines. ALTO
lines come from their ``HPOS``/``VPOS``/``WIDTH``/``HEIGHT`` box and ``String``s; ALTO has no tables or production.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from diffu_page.layout.model import CellRef, Layout, LineKind, Production, Rule, TextLine
from diffu_page.layout.page_xml import (
    PageLine,
    child,
    iter_local,
    local_name,
    parse_points,
    read_page_line,
    reading_order_index,
    region_kind,
)

# PAGE's ``@production`` values, as the two kinds of line a caption tells apart
_PRODUCTION = {
    "printed": Production.PRINTED,
    "typewritten": Production.PRINTED,
    "handwritten-cursive": Production.HANDWRITTEN,
    "handwritten-printscript": Production.HANDWRITTEN,
    "medieval-manuscript": Production.HANDWRITTEN,
    "other": Production.HANDWRITTEN,
}


def _page_size(root: ET.Element) -> tuple[int, int]:
    """The page's size from its ``Page`` element (PAGE ``imageWidth``/``imageHeight`` or ALTO ``WIDTH``/``HEIGHT``).

    Raises:
        ValueError: If the file has no ``Page`` element with a size.
    """
    page = next(iter(iter_local(root, "Page")), None)
    if page is None:
        raise ValueError("no Page element")
    width = page.get("imageWidth") or page.get("WIDTH")
    height = page.get("imageHeight") or page.get("HEIGHT")
    if not (width and height):
        raise ValueError("the Page element has no width and height")
    return int(float(width)), int(float(height))


def _production(element: ET.Element, inherited: Production) -> Production:
    """An element's own PAGE ``@production``, else ``inherited`` (its region's, cell's or table's).

    Raises:
        ValueError: If the value is not one PAGE defines.
    """
    value = element.get("production")
    if value is None:
        return inherited
    if value not in _PRODUCTION:
        raise ValueError(
            f"{local_name(element.tag)} {element.get('id', '')!r}: production {value!r} is not one PAGE defines "
            f"({', '.join(sorted(_PRODUCTION))})"
        )
    return _PRODUCTION[value]


def _text_line(line: PageLine, kind: LineKind, block: str, *, production: Production, cell: CellRef | None) -> TextLine:
    xs = [x for x, _ in line.polygon]
    ys = [y for _, y in line.polygon]
    return TextLine(
        text=line.text,
        box=(min(xs), min(ys), max(xs), max(ys)),
        baseline_y=line.baseline_y,
        kind=kind,
        block=block,
        production=production,
        cell=cell,
    )


def _key(element: ET.Element, kind: str, position: int) -> str:
    """An element's id (PAGE ``@id``, ALTO ``@ID``), else its place among the page's elements of its kind, so that
    elements without an id stay apart."""
    return element.get("id") or element.get("ID") or f"#{kind}{position}"


def _region_lines(
    element: ET.Element,
    kind: LineKind,
    block: str,
    *,
    production: Production = Production.HANDWRITTEN,
    cell: CellRef | None = None,
) -> list[TextLine]:
    """The lines of a PAGE region (direct ``TextLine`` children), top to bottom; ``production`` is what the region's
    container makes them unless the region or a line says otherwise."""
    own = _production(element, production)
    lines = [
        _text_line(read, kind, block, production=_production(line, own), cell=cell)
        for line in element
        if local_name(line.tag) == "TextLine" and (read := read_page_line(line)) is not None
    ]
    return sorted(lines, key=lambda line: line.baseline_y)


def _cell_ref(table: ET.Element, table_key: str, cell: ET.Element) -> CellRef | None:
    """Where a table's child sits: a Transkribus ``TableCell``'s ``@row``/``@col``, a PAGE 2019 ``TextRegion``'s
    ``TableCellRole``; None for any other child.

    Raises:
        ValueError: If a cell lacks its row or its column (both are required in either form).
    """
    match local_name(cell.tag):
        case "TableCell":
            row, column = cell.get("row"), cell.get("col")
        case "TextRegion":
            roles = child(cell, "Roles")
            role = child(roles, "TableCellRole") if roles is not None else None
            if role is None:
                return None
            row, column = role.get("rowIndex"), role.get("columnIndex")
        case _:
            return None
    if row is None or column is None:
        raise ValueError(f"cell {cell.get('id', '')!r} of table {table_key!r} has no row or no column")
    return CellRef(table=table_key, row=int(row), column=int(column))


def _table_lines(table: ET.Element, table_key: str) -> list[TextLine]:
    """A table's lines, cell by cell in file order, each cell's top to bottom."""
    production = _production(table, Production.HANDWRITTEN)
    return [
        line
        for k, cell in enumerate(table)
        if (ref := _cell_ref(table, table_key, cell)) is not None
        for line in _region_lines(cell, LineKind.TABLE, _key(cell, "cell", k), production=production, cell=ref)
    ]


def _page_lines(root: ET.Element) -> list[TextLine]:
    """A PAGE file's lines: text regions in Transkribus reading order, then the tables' cells, each top to bottom."""
    tables = iter_local(root, "TableRegion")
    in_table = {id(element) for table in tables for element in table.iter()}
    text_regions = [element for element in iter_local(root, "TextRegion") if id(element) not in in_table]
    regions = sorted(enumerate(text_regions), key=lambda item: reading_order_index(item[1], item[0]))
    text = [
        line
        for position, element in regions
        for line in _region_lines(element, region_kind(element), _key(element, "region", position))
    ]
    return text + [line for k, table in enumerate(tables) for line in _table_lines(table, _key(table, "table", k))]


def _page_rules(root: ET.Element) -> list[Rule]:
    """The page's ruled lines: every ``SeparatorRegion``'s box."""
    rules = []
    for separator in iter_local(root, "SeparatorRegion"):
        coords = child(separator, "Coords")
        points = parse_points(coords.get("points", "")) if coords is not None else []
        if points:
            xs, ys = [x for x, _ in points], [y for _, y in points]
            rules.append(Rule(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys)))
    return rules


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _alto_baseline_y(line: ET.Element, bottom: float) -> float:
    """The y of an ALTO line's ``BASELINE``: one number (ALTO before 4.2), or the mean y of its points, written
    ``x1,y1 x2,y2`` or ``x1 y1 x2 y2`` (ALTO 4.4's PointsType allows both); ``bottom`` (the box's) without one.

    Raises:
        ValueError: If the baseline is neither one number nor whole x, y pairs.
    """
    numbers = [float(number) for number in _NUMBER.findall(line.get("BASELINE", ""))]
    if not numbers:
        return bottom
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) % 2:
        raise ValueError(f"BASELINE {line.get('BASELINE')!r} is neither one number nor x, y pairs")
    ys = numbers[1::2]
    return sum(ys) / len(ys)


def _alto_line(line: ET.Element, block: str) -> TextLine | None:
    """An ALTO ``TextLine``: its box, its ``String`` contents joined, its ``BASELINE`` (else the box's bottom); None
    for a line without text or without a whole box."""
    text = " ".join(s.get("CONTENT", "") for s in line if local_name(s.tag) == "String").strip()
    box = [line.get(name) for name in ("HPOS", "VPOS", "WIDTH", "HEIGHT")]
    if not text or None in box:
        return None
    x, y, w, h = (float(value) for value in box if value is not None)
    return TextLine(text=text, box=(x, y, x + w, y + h), baseline_y=_alto_baseline_y(line, y + h), block=block)


def _alto_lines(root: ET.Element) -> list[TextLine]:
    """An ALTO file's lines, block by block in file order."""
    lines = (
        _alto_line(line, _key(block, "block", k))
        for k, block in enumerate(iter_local(root, "TextBlock"))
        for line in block
        if local_name(line.tag) == "TextLine"
    )
    return [line for line in lines if line is not None]


def _require_pixels(root: ET.Element) -> None:
    """Refuse an ALTO page whose ``MeasurementUnit`` is not pixels (``mm10``, ``inch1200``): its coordinates would
    be read as pixels.

    Raises:
        ValueError: If the page names another unit.
    """
    unit = next(((e.text or "").strip() for e in root.iter() if local_name(e.tag) == "MeasurementUnit"), "pixel")
    if unit != "pixel":
        raise ValueError(f"the page is measured in {unit}, not pixels")


def load_xml_layout(xml_path: Path) -> Layout:
    """A page's layout from its PAGE or ALTO XML file: its size, its transcribed lines in reading order and its ruled
    lines.

    Raises:
        ValueError: If the file is not well-formed XML, its ``Page`` element is missing or has no size, an ALTO page is
            measured in other units than pixels, a line's production is not one PAGE defines, or a table cell has no
            row or column; the message names the file.
    """
    try:
        return _layout(ET.parse(xml_path).getroot())
    except (ET.ParseError, ValueError) as error:
        raise ValueError(f"{xml_path}: {error}") from error


def _layout(root: ET.Element) -> Layout:
    width, height = _page_size(root)
    if local_name(root.tag) == "alto":
        _require_pixels(root)
        return Layout(width=width, height=height, lines=_alto_lines(root))
    return Layout(width=width, height=height, lines=_page_lines(root), rules=_page_rules(root))
