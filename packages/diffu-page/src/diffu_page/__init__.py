"""Full-page synthesis from Diffu-generated lines.

A *module*, not a model — no new training. It turns Diffu's per-line output into full, well-blended
document pages for HTR/OCR data, in arbitrary layouts (single/multi-column, semi-tables, spreads).
Three stages with a clean seam between them::

    layout  →  list[LineBox]  →  render  →  ink rasters  →  compose  →  page image + labels

- ``layout``  : carve the page into regions and flow text into line boxes (coords + text per line).
                Source the boxes from a parameterized template (:func:`build_layout`) or from real
                ALTO/PAGE-XML (:func:`load_xml_layout`).
- ``render``  : one (text, style) → an ink-on-transparent line raster. :class:`DiffuRenderer` wraps
                ``generate.generate_line``; :class:`PlaceholderRenderer` draws gray ink so the layout
                + compositor run with no model and no GPU.
- ``compose`` : seat each line on a (preferably real) paper background — baseline-aligned, the model's
                verbatim ink multiplied *through* the page (halo-excluding matte → no shadow), draw
                furniture, then ONE global pass (lighting, warm aging, degradation, JPEG, warp) so the
                page reads as a single scan across the real pristine→filthy range. Emits polygon +
                transcription labels. Real backgrounds via :func:`diffu_page.paper.extract_paper_crops`.

CLI: ``python -m diffu_page`` (or ``diffu-page``). Imaging is OpenCV; torch is imported lazily only
inside :class:`DiffuRenderer`, so the pipeline runs on a light CPU box with just the ``page`` extra.
"""

from __future__ import annotations

from .augment import AugmentConfig
from .compose import compose_page, compose_page_from_inks
from .labels import (
    EXPORTERS,
    FORMATS,
    LineLabel,
    PageLabels,
    WordLabel,
    export,
    to_alto,
    to_json,
    to_page_xml,
)
from .layout import (
    Column,
    Layout,
    LineBox,
    PageSpec,
    Rule,
    build_layout,
    load_xml_layout,
    validate_layout,
)
from .paper import paper_background
from .render import DiffuRenderer, ErukuRenderer, LineRenderer, PlaceholderRenderer
from .validate import MatchResult

__all__ = [
    "LineBox",
    "Rule",
    "Column",
    "PageSpec",
    "Layout",
    "build_layout",
    "load_xml_layout",
    "validate_layout",
    "LineRenderer",
    "PlaceholderRenderer",
    "DiffuRenderer",
    "ErukuRenderer",
    "compose_page",
    "compose_page_from_inks",
    "AugmentConfig",
    "paper_background",
    "PageLabels",
    "LineLabel",
    "WordLabel",
    "export",
    "EXPORTERS",
    "FORMATS",
    "to_json",
    "to_page_xml",
    "to_alto",
    "MatchResult",
]
