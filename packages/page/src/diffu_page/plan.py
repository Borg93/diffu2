"""Plan a page layout as FLUX.3 crops: the windows, prompts and line boxes the v2 LoRA was trained on.

A :class:`diffu_page.layout.Layout` (a real PAGE or ALTO page, refilled with new text) becomes the v2 crop builder's
lines, so the crops come out of the same windowing (whole lines, about 1 megapixel at a 64 px line pitch, NeMo bucket
sizes) and the same caption grammar as the training crops. Table cells are left out, as the builder leaves them out. The style is a ``Provenance`` (archive,
years) and a ``ScanLook`` (scan, paper, ink), as in every v2 caption. CPU only.
"""

from __future__ import annotations

from operator import attrgetter

from pydantic import BaseModel

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import Caption, LineRecord, caption_for, caption_order
from diffu_page.captions.provenance import Provenance
from diffu_page.crops.geometry import Crop, CropSettings, Line, Region, crops_for_page
from diffu_page.layout import ROLE_NAMES, TABLE, Layout, LineBox


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
    lines: list[LineRecord]  # boxes on 0-1000 of the crop, y first, in the caption's numbering
    reading_order: list[int]  # indices into ``lines``: block by block in page order, top to bottom in each


def _source(box: LineBox) -> str:
    """The block a line came from, as reading order tracks it."""
    return f"{box.region}|{box.block}"


def _line(box: LineBox) -> Line:
    x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
    return Line(
        polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        baseline_y=box.baseline_y,
        text=box.text,
        role=box.region if box.region in ROLE_NAMES else "",
        source=_source(box),
    )


def regions_of(layout: Layout) -> list[Region]:
    """The layout's text lines grouped by region and source block, in reading order, each group's lines top to
    bottom; table cells are left out."""
    grouped: dict[tuple[str, str], list[Line]] = {}
    for box in layout.boxes:
        if box.text.strip() and box.region != TABLE:
            grouped.setdefault((box.region, box.block), []).append(_line(box))
    return [Region(lines=sorted(lines, key=attrgetter("baseline_y"))) for lines in grouped.values()]


def _reading_order(crop: Crop, block_rank: dict[str, int]) -> list[int]:
    """The crop's lines (in caption numbering) in reading order: by their block's place on the page, then down."""
    numbered = caption_order(crop.lines)
    return sorted(range(len(numbered)), key=lambda i: (block_rank[numbered[i].source], numbered[i].baseline_y))


def _plan(crop: Crop, caption: Caption, block_rank: dict[str, int]) -> CropPlan:
    return CropPlan(
        rect=crop.rect,
        width=crop.size[0],
        height=crop.size[1],
        caption=caption.text,
        lines=caption.lines,
        reading_order=_reading_order(crop, block_rank),
    )


def plan_page(layout: Layout, style: PageStyle, settings: CropSettings | None = None) -> list[CropPlan]:
    """The crops the v2 builder would cut from this page, each with its caption in ``style``."""
    crops = crops_for_page(regions_of(layout), (layout.width, layout.height), settings or CropSettings())
    block_rank = {source: rank for rank, source in enumerate(dict.fromkeys(_source(box) for box in layout.boxes))}
    return [_plan(crop, caption_for(crop, style.provenance, style.look), block_rank) for crop in crops]
