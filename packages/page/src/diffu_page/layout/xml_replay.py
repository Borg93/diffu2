"""A page layout from a real PAGE or ALTO XML file.

PAGE lines are read with :mod:`diffu_page.layout.page_xml`, the reader the training-crop builder uses, so a page is
planned for generation exactly as its training crops were cut: the same outlines, baselines, region roles and
Transkribus reading order. ALTO lines come from their ``HPOS``/``VPOS``/``WIDTH``/``HEIGHT`` box and ``String``s.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from diffu_page.layout.model import RUNNING_TEXT, TABLE, Layout, LineBox
from diffu_page.layout.page_xml import (
    PageLine,
    iter_local,
    local_name,
    parse_points,
    read_page_line,
    reading_order_index,
    region_role,
)

_TABLE_CONTAINERS = ("TableRegion", "TableCell")


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


def _box(line: PageLine, region: str, block: str) -> LineBox:
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
    )


def _region_lines(element: ET.Element, region: str) -> list[LineBox]:
    """The lines of a PAGE region (direct ``TextLine`` children), top to bottom."""
    read = (read_page_line(line) for line in element if local_name(line.tag) == "TextLine")
    boxes = [_box(line, region, element.get("id", "")) for line in read if line is not None]
    return sorted(boxes, key=lambda box: box.baseline_y)


def _page_lines(root: ET.Element) -> list[LineBox]:
    """A PAGE file's lines: text regions in Transkribus reading order, then table cells, each top to bottom."""
    regions = sorted(enumerate(iter_local(root, "TextRegion")), key=lambda item: reading_order_index(item[1], item[0]))
    text = [line for _, element in regions for line in _region_lines(element, region_role(element) or RUNNING_TEXT)]
    cells = [line for cell in iter_local(root, "TableCell") for line in _region_lines(cell, TABLE)]
    return text + cells


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
    """A page's layout from its PAGE or ALTO XML file: its size and its transcribed lines in reading order.

    Raises:
        ValueError: If the file's ``Page`` element is missing or has no size.
    """
    root = ET.parse(xml_path).getroot()
    width, height = _page_size(root)
    is_alto = local_name(root.tag) == "alto"
    return Layout(width=width, height=height, boxes=_alto_lines(root) if is_alto else _page_lines(root))
