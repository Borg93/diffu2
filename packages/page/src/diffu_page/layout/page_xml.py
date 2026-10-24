"""Reading PAGE XML, namespace-agnostic: a line's text, outline and baseline, a region's type and reading order. The
pieces :mod:`diffu_page.layout.xml_replay` reads a page's layout with.
"""

from __future__ import annotations

import re
import statistics
import xml.etree.ElementTree as ET

from pydantic import BaseModel

from diffu_page.layout.model import ROLE_NAMES

_STRUCTURE = re.compile(r"structure \{type:([^;}]+)")  # Transkribus writes a region's type into @custom
_READING_ORDER = re.compile(r"readingOrder \{index:(\d+);")


class PageLine(BaseModel):
    """A PAGE ``TextLine`` as read: its transcription, its outline and the height of its baseline (page px)."""

    text: str
    polygon: list[tuple[float, float]]
    baseline_y: float


def local_name(tag: str) -> str:
    """An XML tag without its ``{namespace}``, so PAGE and ALTO match across schema versions."""
    return tag.rsplit("}", 1)[-1]


def iter_local(elem: ET.Element, name: str) -> list[ET.Element]:
    """Every element under ``elem`` (itself included) whose local tag is ``name``."""
    return [e for e in elem.iter() if local_name(e.tag) == name]


def child(element: ET.Element, name: str) -> ET.Element | None:
    """The first direct child of ``element`` whose local tag is ``name``."""
    return next((c for c in element if local_name(c.tag) == name), None)


def parse_points(points: str) -> list[tuple[float, float]]:
    """Parse a PAGE ``points="x1,y1 x2,y2 …"`` string into a list of (x, y) vertices."""
    pts: list[tuple[float, float]] = []
    for pair in points.split():
        if "," in pair:
            x, y = pair.split(",")[:2]
            pts.append((float(x), float(y)))
    return pts


def _own_text(element: ET.Element) -> str:
    """The Unicode of ``element``'s own ``TextEquiv`` (not of its children's), or ''."""
    text_equiv = child(element, "TextEquiv")
    unicode = child(text_equiv, "Unicode") if text_equiv is not None else None
    return (unicode.text or "") if unicode is not None else ""


def page_line_text(line: ET.Element) -> str:
    """A PAGE TextLine's transcription: its own TextEquiv/Unicode, else its Words' joined."""
    own = _own_text(line)
    if own:
        return own
    return " ".join(text for word in line if local_name(word.tag) == "Word" and (text := _own_text(word)))


def read_page_line(line: ET.Element) -> PageLine | None:
    """A PAGE ``TextLine`` from its own children: the transcription, the ``Coords`` outline (never a Word's) and the
    baseline height (the mean of its ``Baseline`` points, else the outline's lowest point); None without text or an
    outline of at least three points."""
    text = page_line_text(line).strip()
    coords = child(line, "Coords")
    if not text or coords is None:
        return None
    polygon = parse_points(coords.get("points", ""))
    if len(polygon) < 3:
        return None
    baseline = child(line, "Baseline")
    points = parse_points(baseline.get("points", "")) if baseline is not None else []
    baseline_y = statistics.mean(y for _, y in points) if points else max(y for _, y in polygon)
    return PageLine(text=text, polygon=polygon, baseline_y=baseline_y)


def region_type(region: ET.Element) -> str:
    """A ``TextRegion``'s PAGE type, lower case: its ``@type``, else the ``structure {type:...}`` Transkribus writes
    into its ``@custom``; '' when it has neither."""
    structure = _STRUCTURE.search(region.get("custom") or "")
    return (region.get("type") or (structure.group(1) if structure else "")).lower()


def region_role(region: ET.Element) -> str:
    """The role a ``TextRegion`` gives its lines: its PAGE type when that names one (see ``ROLE_NAMES``), else ''."""
    kind = region_type(region)
    return kind if kind in ROLE_NAMES else ""


def reading_order_index(region: ET.Element, position: int) -> int:
    """A region's Transkribus reading-order index (``readingOrder {index:N;}`` in ``@custom``), else ``position``."""
    match = _READING_ORDER.search(region.get("custom") or "")
    return int(match.group(1)) if match else position
