"""Parse PAGE-XML and ALTO-XML into per-line records. Namespace-agnostic (matches local tag names),
so it works across PAGE/ALTO versions. No torch needed.
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


def local_name(tag: str) -> str:
    """An XML tag without its ``{namespace}``, so PAGE and ALTO match across schema versions."""
    return tag.rsplit("}", 1)[-1]


def iter_local(elem: ET.Element, name: str) -> list[ET.Element]:
    """Every element under ``elem`` (itself included) whose local tag is ``name``."""
    return [e for e in elem.iter() if local_name(e.tag) == name]


def parse_points(points: str) -> list[tuple[float, float]]:
    """Parse a PAGE ``points="x1,y1 x2,y2 …"`` string into a list of (x, y) vertices."""
    pts: list[tuple[float, float]] = []
    for pair in points.split():
        if "," in pair:
            x, y = pair.split(",")[:2]
            pts.append((float(x), float(y)))
    return pts


def page_line_text(line: ET.Element) -> str:
    """A PAGE TextLine's transcription: its own TextEquiv/Unicode, else its Words' joined."""
    for te in line:
        if local_name(te.tag) == "TextEquiv":
            uni = next((u for u in te if local_name(u.tag) == "Unicode"), None)
            if uni is not None and uni.text:
                return uni.text
    words: list[str] = []
    for w in line:
        if local_name(w.tag) == "Word":
            for te in w:
                if local_name(te.tag) == "TextEquiv":
                    uni = next((u for u in te if local_name(u.tag) == "Unicode"), None)
                    if uni is not None and uni.text:
                        words.append(uni.text)
    return " ".join(words)


def _page_records(root: ET.Element, path: str) -> list[LineRecord]:
    records: list[LineRecord] = []
    for page in iter_local(root, "Page"):
        img = page.attrib.get("imageFilename", "")
        src = os.path.splitext(os.path.basename(img or path))[0]
        for line in iter_local(page, "TextLine"):
            coords = next((e for e in line if local_name(e.tag) == "Coords"), None)
            pts = parse_points(coords.attrib.get("points", "")) if coords is not None else []
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            text = page_line_text(line)
            if text:
                poly = pts if len(pts) >= 3 else None
                records.append(LineRecord(image=img, bbox=bbox, text=text, source_id=src, polygon=poly))
    return records


def _alto_records(root: ET.Element, path: str) -> list[LineRecord]:
    fname = next((e.text for e in root.iter() if local_name(e.tag) == "fileName" and e.text), "")
    src = os.path.splitext(os.path.basename(fname or path))[0]
    records: list[LineRecord] = []
    for line in iter_local(root, "TextLine"):
        a = line.attrib
        try:
            x, y, w, h = float(a["HPOS"]), float(a["VPOS"]), float(a["WIDTH"]), float(a["HEIGHT"])
        except KeyError:
            continue
        strings = [s.attrib.get("CONTENT", "") for s in line if local_name(s.tag) == "String"]
        text = " ".join(s for s in strings if s)
        if text:
            records.append(LineRecord(image=fname, bbox=(x, y, x + w, y + h), text=text, source_id=src))
    return records


def parse_file(path: str) -> list[LineRecord]:
    """Line records of a PAGE or ALTO file (auto-detected from the root tag; the file is parsed once)."""
    root = ET.parse(path).getroot()
    if local_name(root.tag).lower() == "alto":
        return _alto_records(root, path)
    return _page_records(root, path)  # PcGts / PAGE
