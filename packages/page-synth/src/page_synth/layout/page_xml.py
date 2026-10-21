"""Namespace-agnostic PAGE/ALTO XML reading: elements by local tag, Coords/Baseline points, a TextLine's transcription.

Shared by the layout replay (:mod:`page_synth.layout.xml_replay`) and the training-crop builder
(:mod:`page_synth.data.build_crops`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET


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
