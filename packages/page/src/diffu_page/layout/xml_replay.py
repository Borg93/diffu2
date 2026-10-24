"""A page layout from a real PAGE or ALTO XML file.

PAGE lines are read with :mod:`diffu_page.layout.page_xml`, the reader the training-crop builder uses, so a page is
planned for generation exactly as its training crops were cut: the same outlines, baselines, region roles and
Transkribus reading order. A PAGE table is read in either form a template comes in: Transkribus' ``TableCell``
(``@row``, ``@col``) or PAGE 2019's nested ``TextRegion`` with a ``TableCellRole``. A line's ``@production`` (else its
region's, its cell's or its table's) says whether it is printed; ``SeparatorRegion``s are the page's ruled lines. ALTO
lines come from their ``HPOS``/``VPOS``/``WIDTH``/``HEIGHT`` box and ``String``s; ALTO has no tables or production.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from diffu_page.layout.model import RUNNING_TEXT, TABLE, CellRef, Layout, LineBox, Production, Rule
from diffu_page.layout.page_xml import (
    PageLine,
    child,
    iter_local,
    local_name,
    parse_points,
    read_page_line,
    reading_order_index,
    region_role,
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


def _box(line: PageLine, region: str, block: str, *, production: Production, cell: CellRef | None) -> LineBox:
    xs = [x for x, _ in line.polygon]
    ys = [y for _, y in line.polygon]
    return LineBox(
        text=line.text,
        x=min(xs),
        y=min(ys),
        w=max(xs) - min(xs),
        h=max(ys) - min(ys),
        baseline_y=line.baseline_y,
        region=region,
        block=block,
        production=production,
        cell=cell,
    )


def _region_lines(
    element: ET.Element, region: str, *, production: Production = Production.HANDWRITTEN, cell: CellRef | None = None
) -> list[LineBox]:
    """The lines of a PAGE region (direct ``TextLine`` children), top to bottom; ``production`` is what the region's
    container makes them unless the region or a line says otherwise."""
    own = _production(element, production)
    boxes = [
        _box(read, region, element.get("id", ""), production=_production(line, own), cell=cell)
        for line in element
        if local_name(line.tag) == "TextLine" and (read := read_page_line(line)) is not None
    ]
    return sorted(boxes, key=lambda box: box.baseline_y)


def _cell_ref(table: ET.Element, cell: ET.Element) -> CellRef | None:
    """Where a table's child sits: a Transkribus ``TableCell``'s ``@row``/``@col``, a PAGE 2019 ``TextRegion``'s
    ``TableCellRole``; None for any other child.

    Raises:
        ValueError: If a ``TableCell`` has no ``@row`` or ``@col``.
    """
    match local_name(cell.tag):
        case "TableCell":
            row, column = cell.get("row"), cell.get("col")
            if row is None or column is None:
                raise ValueError(f"TableCell {cell.get('id', '')!r} of table {table.get('id', '')!r} has no row or col")
            return CellRef(table=table.get("id", ""), row=int(row), column=int(column))
        case "TextRegion":
            roles = child(cell, "Roles")
            role = child(roles, "TableCellRole") if roles is not None else None
            if role is None:
                return None
            return CellRef(
                table=table.get("id", ""), row=int(role.get("rowIndex", 0)), column=int(role.get("columnIndex", 0))
            )
        case _:
            return None


def _table_lines(table: ET.Element) -> list[LineBox]:
    """A table's lines, cell by cell in file order, each cell's top to bottom."""
    production = _production(table, Production.HANDWRITTEN)
    return [
        line
        for cell in table
        if (ref := _cell_ref(table, cell)) is not None
        for line in _region_lines(cell, TABLE, production=production, cell=ref)
    ]


def _page_lines(root: ET.Element) -> list[LineBox]:
    """A PAGE file's lines: text regions in Transkribus reading order, then the tables' cells, each top to bottom."""
    tables = iter_local(root, "TableRegion")
    in_table = {id(element) for table in tables for element in table.iter()}
    text_regions = [element for element in iter_local(root, "TextRegion") if id(element) not in in_table]
    regions = sorted(enumerate(text_regions), key=lambda item: reading_order_index(item[1], item[0]))
    text = [line for _, element in regions for line in _region_lines(element, region_role(element) or RUNNING_TEXT)]
    return text + [line for table in tables for line in _table_lines(table)]


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


def _alto_line(line: ET.Element, block: str) -> LineBox | None:
    """An ALTO ``TextLine``: its box, its ``String`` contents joined, its ``BASELINE`` (else the box's bottom)."""
    text = " ".join(s.get("CONTENT", "") for s in line if local_name(s.tag) == "String").strip()
    if not text or line.get("HPOS") is None:
        return None
    x, y = float(line.get("HPOS", 0)), float(line.get("VPOS", 0))
    w, h = float(line.get("WIDTH", 0)), float(line.get("HEIGHT", 0))
    baseline = line.get("BASELINE", "")
    points = parse_points(baseline)
    baseline_y = sum(py for _, py in points) / len(points) if points else float(baseline or y + h)
    return LineBox(text=text, x=x, y=y, w=w, h=h, baseline_y=baseline_y, block=block)


def _alto_lines(root: ET.Element) -> list[LineBox]:
    """An ALTO file's lines, block by block in file order."""
    lines = (
        _alto_line(line, block.get("ID", ""))
        for block in iter_local(root, "TextBlock")
        for line in block
        if local_name(line.tag) == "TextLine"
    )
    return [line for line in lines if line is not None]


def load_xml_layout(xml_path: Path) -> Layout:
    """A page's layout from its PAGE or ALTO XML file: its size, its transcribed lines in reading order and its ruled
    lines.

    Raises:
        ValueError: If the file's ``Page`` element is missing or has no size, a line's production is not one PAGE
            defines, or a table cell has no row or column.
    """
    root = ET.parse(xml_path).getroot()
    width, height = _page_size(root)
    if local_name(root.tag) == "alto":
        return Layout(width=width, height=height, boxes=_alto_lines(root))
    return Layout(width=width, height=height, boxes=_page_lines(root), rules=_page_rules(root))
