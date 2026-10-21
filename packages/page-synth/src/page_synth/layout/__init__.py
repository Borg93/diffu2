"""Page layouts: carve a page into regions and lines, from a real ALTO/PAGE page or a parameterized template.

A :class:`Layout` is the page's line boxes (coordinates, text, region, block); :func:`load_xml_layout` replays a
real page, :func:`build_layout` draws a template from a :class:`PageSpec`, :func:`refill` writes new text into a
layout's running-text lines. Pure geometry, no imaging.
"""

from page_synth.layout.edits import is_body
from page_synth.layout.model import (
    Align,
    Column,
    Layout,
    LineBox,
    Measure,
    PageSpec,
    Rule,
    TableColumn,
    validate_layout,
)
from page_synth.layout.templates import TEMPLATES, LeafPrefer, build_layout, to_single_leaf
from page_synth.layout.text_refill import refill, running_text
from page_synth.layout.xml_replay import load_xml_layout

__all__ = [
    "TEMPLATES",
    "Align",
    "Column",
    "Layout",
    "LeafPrefer",
    "LineBox",
    "Measure",
    "PageSpec",
    "Rule",
    "TableColumn",
    "build_layout",
    "is_body",
    "load_xml_layout",
    "refill",
    "running_text",
    "to_single_leaf",
    "validate_layout",
]
