"""An ALTO file read for what a check on its labels needs and :mod:`diffu_page`'s layout reader leaves out: who wrote
it, the page's confidence, and each line's outline."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from diffu_page.page.layout.page_xml import local_name

type Point = tuple[float, float]
type Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in the scan's pixels

WRITERS = ("htrflow", "Transkribus")  # the tools a pool's files name: the old pipeline, and a transcriber's export


class AltoLine(BaseModel):
    """A text line: its text (its words joined by a space), its box and its outline (its box's corners where the file
    gives no polygon)."""

    model_config = ConfigDict(frozen=True)

    text: str
    box: Box
    outline: tuple[Point, ...]

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


class AltoPage(BaseModel):
    """A scan's labels: the tool that wrote them (``""`` if none of :data:`WRITERS`), the scan's size, the confidence
    the tool gave the page (None if it gave none), how many regions it found, its lines that hold text, in the file's
    order, and how many lines hold none."""

    model_config = ConfigDict(frozen=True)

    writer: str
    width: float
    height: float
    confidence: float | None
    regions: int
    lines: tuple[AltoLine, ...]
    empty_lines: int


def _number(element: ET.Element, name: str) -> float:
    try:
        return float(element.get(name, 0))
    except ValueError:
        return 0.0


def _points(text: str) -> tuple[Point, ...]:
    numbers = [float(n) for n in text.replace(",", " ").split()]
    return tuple(zip(numbers[0::2], numbers[1::2], strict=False))


def _line(element: ET.Element) -> AltoLine | None:
    text = " ".join(child.get("CONTENT", "") for child in element if local_name(child.tag) == "String").strip()
    if not text:
        return None
    x, y = _number(element, "HPOS"), _number(element, "VPOS")
    box = (x, y, x + _number(element, "WIDTH"), y + _number(element, "HEIGHT"))
    polygon = next((p for p in element.iter() if local_name(p.tag) == "Polygon"), None)
    outline = _points(polygon.get("POINTS", "")) if polygon is not None else ()
    if len(outline) < 3:
        outline = ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))
    return AltoLine(text=text, box=box, outline=outline)


def read_alto(path: Path) -> AltoPage:
    """The labels in the ALTO file at ``path``.

    Raises:
        ValueError: If the file cannot be read, is not well-formed XML or has no ``Page``; the message names the file.
    """
    try:
        source = path.read_text(encoding="utf-8")
        root = ET.fromstring(source)
    except (OSError, UnicodeDecodeError, ET.ParseError) as error:
        raise ValueError(f"{path}: {error}") from error
    page = next((e for e in root.iter() if local_name(e.tag) == "Page"), None)
    if page is None:
        raise ValueError(f"{path}: no Page")
    found = [(_line(e)) for e in page.iter() if local_name(e.tag) == "TextLine"]
    confidence = page.get("PC")
    return AltoPage(
        writer=next((w for w in WRITERS if w in source[:4000]), ""),
        width=_number(page, "WIDTH"),
        height=_number(page, "HEIGHT"),
        confidence=float(confidence) if confidence else None,
        regions=sum(1 for e in page.iter() if local_name(e.tag) == "TextBlock"),
        lines=tuple(line for line in found if line is not None),
        empty_lines=sum(1 for line in found if line is None),
    )
