"""Page layouts: a real PAGE or ALTO page read as its size, its text lines in reading order and its ruled lines
(:func:`load_xml_layout`), a scan's pages, a two-page opening split at its gutter (:func:`scan_pages`), and new text
written into its running-text lines (:func:`refill`)."""

from diffu_page.layout.model import ROLE_NAMES, CellRef, Layout, LineKind, PageBox, Production, Rule, TextLine
from diffu_page.layout.pages import PageSide, ScanPage, gutter, is_unsplit_spread, scan_pages
from diffu_page.layout.text_refill import refill, running_text
from diffu_page.layout.xml_replay import load_xml_layout

__all__ = [
    "ROLE_NAMES",
    "CellRef",
    "Layout",
    "LineKind",
    "PageBox",
    "PageSide",
    "Production",
    "Rule",
    "ScanPage",
    "TextLine",
    "gutter",
    "is_unsplit_spread",
    "load_xml_layout",
    "refill",
    "running_text",
    "scan_pages",
]
