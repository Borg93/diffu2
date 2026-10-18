"""Parse PAGE-XML and ALTO-XML into per-line records. Namespace-agnostic (matches local tag names),
so it works across PAGE/ALTO versions. Pure stdlib — no torch needed.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from pydantic import BaseModel


class LineRecord(BaseModel):
    image: str  # page image filename as referenced in the XML
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    text: str
    source_id: str  # page/document id (writer proxy)
    polygon: list[tuple[float, float]] | None = None  # PAGE <Coords> line polygon (None for ALTO rects)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]  # strip "{namespace}" -> local name


def _iter_local(elem: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in elem.iter() if _local(e.tag) == name]


def _parse_points(points: str) -> list[tuple[float, float]]:
    """Parse a PAGE ``points="x1,y1 x2,y2 …"`` string into a list of (x, y) vertices."""
    pts: list[tuple[float, float]] = []
    for pair in points.split():
        if "," in pair:
            x, y = pair.split(",")[:2]
            pts.append((float(x), float(y)))
    return pts


def _page_line_text(line: ET.Element) -> str:
    # Prefer the TextLine-level TextEquiv/Unicode; else join Word-level Unicodes.
    for te in line:
        if _local(te.tag) == "TextEquiv":
            uni = next((u for u in te if _local(u.tag) == "Unicode"), None)
            if uni is not None and uni.text:
                return uni.text
    words: list[str] = []
    for w in line:
        if _local(w.tag) == "Word":
            for te in w:
                if _local(te.tag) == "TextEquiv":
                    uni = next((u for u in te if _local(u.tag) == "Unicode"), None)
                    if uni is not None and uni.text:
                        words.append(uni.text)
    return " ".join(words)


def parse_page_xml(path: str) -> list[LineRecord]:
    root = ET.parse(path).getroot()
    records: list[LineRecord] = []
    for page in _iter_local(root, "Page"):
        img = page.attrib.get("imageFilename", "")
        src = os.path.splitext(os.path.basename(img or path))[0]
        for line in _iter_local(page, "TextLine"):
            coords = next((e for e in line if _local(e.tag) == "Coords"), None)
            pts = _parse_points(coords.attrib.get("points", "")) if coords is not None else []
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            text = _page_line_text(line)
            if text:
                poly = pts if len(pts) >= 3 else None
                records.append(LineRecord(image=img, bbox=bbox, text=text, source_id=src, polygon=poly))
    return records


def parse_alto_xml(path: str) -> list[LineRecord]:
    root = ET.parse(path).getroot()
    fname = next((e.text for e in root.iter() if _local(e.tag) == "fileName" and e.text), "")
    src = os.path.splitext(os.path.basename(fname or path))[0]
    records: list[LineRecord] = []
    for line in _iter_local(root, "TextLine"):
        a = line.attrib
        try:
            x, y, w, h = float(a["HPOS"]), float(a["VPOS"]), float(a["WIDTH"]), float(a["HEIGHT"])
        except KeyError:
            continue
        strings = [s.attrib.get("CONTENT", "") for s in line if _local(s.tag) == "String"]
        text = " ".join(s for s in strings if s)
        if text:
            records.append(LineRecord(image=fname, bbox=(x, y, x + w, y + h), text=text, source_id=src))
    return records


def parse_file(path: str) -> list[LineRecord]:
    """Auto-detect PAGE vs ALTO from the root tag."""
    root = ET.parse(path).getroot()
    if _local(root.tag).lower() == "alto":
        return parse_alto_xml(path)
    return parse_page_xml(path)  # PcGts / PAGE
