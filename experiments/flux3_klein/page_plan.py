"""Plan a diffu-page layout as FLUX.3 crops: the windows, prompts and line boxes the v2 LoRA was trained on.

A :class:`diffu_page.Layout` (a real ALTO/PAGE page replayed and refilled, or a synthetic template) becomes the v2 crop
builder's lines, so the crops come out of the same windowing (whole lines, about 1 megapixel at a 64 px line pitch,
NeMo bucket sizes) and the same caption grammar as the training crops. The style is a ``Provenance`` (archive,
years) and a ``ScanLook`` (scan, paper, ink), as in every v2 caption. CPU only.
"""

from __future__ import annotations

from build_layout_crops import (
    Caption,
    Crop,
    CropSettings,
    Line,
    LineRecord,
    Provenance,
    Region,
    ScanLook,
    caption_for,
    crops_for_page,
)
from diffu_page import Layout, LineBox
from pydantic import BaseModel

# diffu-page region -> the PAGE type the v2 captions name as a role; running text and table cells name none
ROLES = {
    "header": "heading",
    "reg_header": "heading",
    "rubric": "heading",
    "table_header": "heading",
    "marginalia": "marginalia",
    "margin_key": "marginalia",
    "page_number": "page-number",
    "catchword": "catch-word",
    "signature": "signature-mark",
}


class PageStyle(BaseModel):
    """The style a page is drawn in, named in every crop's caption."""

    provenance: Provenance
    look: ScanLook


class CropPlan(BaseModel):
    """One crop of a page to draw: where it sits on the page, its size and its prompt."""

    rect: tuple[float, float, float, float]  # page px: x0, y0, x1, y1
    width: int
    height: int
    caption: str
    lines: list[LineRecord]  # boxes on 0-1000 of the crop, y first


def _line(box: LineBox) -> Line:
    x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
    return Line(
        polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        baseline_y=box.baseline_y,
        text=box.text,
        role=ROLES.get(box.region, ""),
    )


def regions_of(layout: Layout) -> list[Region]:
    """The layout's lines grouped by region and source block (a replayed page's text regions, a template's
    columns), in reading order, each group's lines top to bottom."""
    grouped: dict[tuple[str, str], list[Line]] = {}
    for box in layout.boxes:
        if box.text.strip():
            grouped.setdefault((box.region, box.block), []).append(_line(box))
    return [Region(lines=sorted(lines, key=lambda line: line.baseline_y)) for lines in grouped.values()]


def _plan(crop: Crop, caption: Caption) -> CropPlan:
    return CropPlan(
        rect=crop.rect, width=crop.size[0], height=crop.size[1], caption=caption.text, lines=caption.lines
    )


def plan_page(layout: Layout, style: PageStyle, settings: CropSettings | None = None) -> list[CropPlan]:
    """The crops the v2 builder would cut from this page, each with its caption in ``style``."""
    crops = crops_for_page(
        regions_of(layout), (layout.spec.width, layout.spec.height), settings or CropSettings()
    )
    return [_plan(crop, caption_for(crop, style.provenance, style.look)) for crop in crops]
