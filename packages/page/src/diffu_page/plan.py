"""Plan a page layout as FLUX.3 crops: the windows, prompts and line boxes the LoRAs were trained on.

A :class:`diffu_page.layout.Layout` (a real PAGE or ALTO page, refilled with new text) becomes the v2 crop builder's
lines, so the crops come out of the same windowing (whole lines, about 1 megapixel at a 64 px line pitch, NeMo bucket
sizes) and the same caption grammar as the training crops. A table is planned as one region, so its windows hold whole
rows across it, and the page's ruled lines are kept in the windows they cross; a page without tables, printed lines or
ruled lines is planned exactly as before.
The style is a ``Provenance`` (archive, years) and a ``ScanLook`` (scan, paper, ink), as in every v2 caption. A page
drawn from a :class:`Reference` (style by example) is captioned as the reference pairs were (v3 on): its style's
words give way to the reference, in the caption form the LoRA in use was trained on. CPU only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from diffu_page.captions.hand import ReferenceLook, measure_reference
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import Caption, LineRecord, caption_for, caption_order
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.records import read_records
from diffu_page.captions.restyle import CaptionParts, ReferenceCaption, Style, reference_caption
from diffu_page.crops.geometry import Crop, CropSettings, LayoutBox, crops_for_page, regions_of
from diffu_page.layout import Layout


class PageStyle(BaseModel):
    """The style a page is drawn in, named in every crop's caption."""

    provenance: Provenance
    look: ScanLook


class Reference(BaseModel):
    """A crop whose hand, ink and paper a page is drawn in (style by example), and the caption form the LoRA in use
    was trained on. A described caption states the reference's hand as measured inside its line boxes, so it needs
    them; :meth:`of_crop` reads them for a crop of ``diffu_page.data.build_crops``."""

    image: Path
    caption: ReferenceCaption = ReferenceCaption.PLAIN
    lines: list[LayoutBox] = []  # the reference's line boxes, 0-1000 of the image, y first

    @model_validator(mode="after")
    def _measurable(self) -> Self:
        if self.caption is ReferenceCaption.DESCRIBED and not self.lines:
            raise ValueError(
                f"a described caption measures the reference's hand in its line boxes: {self.image} has none"
            )
        return self

    @classmethod
    def of_crop(cls, image: Path, caption: ReferenceCaption = ReferenceCaption.PLAIN) -> Reference:
        """A crop ``diffu_page.data.build_crops`` wrote, its line boxes read from the ``metadata.jsonl`` beside it.

        Raises:
            ValueError: If that metadata has no record of the crop.
        """
        metadata = image.parent / "metadata.jsonl"
        record = next((record for record in read_records(metadata) if record.file_name == image.name), None)
        if record is None:
            raise ValueError(f"{metadata} has no record of {image.name}")
        return cls(image=image, caption=caption, lines=[line.bbox for line in record.lines])

    def look(self) -> ReferenceLook | None:
        """The reference's look as its captions state it: measured for a described caption, else None."""
        return measure_reference(self.image, self.lines) if self.caption is ReferenceCaption.DESCRIBED else None


class CropPlan(BaseModel):
    """One crop of a page to draw: where it sits on the page, its size and its prompt."""

    rect: tuple[float, float, float, float]  # page px: x0, y0, x1, y1
    width: int
    height: int
    caption: str
    lines: list[LineRecord]  # boxes on 0-1000 of the crop, y first, in the caption's numbering
    reading_order: list[int]  # indices into ``lines``: block by block in page order, top to bottom in each
    rules: list[LayoutBox] = Field(default_factory=list)  # the ruled lines' boxes, 0-1000 of the crop, y first

    @model_validator(mode="after")
    def _reading_order_orders_the_lines(self) -> Self:
        """A plan read from a job file fails here, before its crop is drawn, not at the label export after it."""
        if sorted(self.reading_order) != list(range(len(self.lines))):
            raise ValueError(
                f"reading order {self.reading_order} is not an order of the plan's {len(self.lines)} lines"
            )
        return self


def _reading_order(crop: Crop, block_rank: dict[str, int]) -> list[int]:
    """The crop's lines (in caption numbering) in reading order: by their block's place on the page, then down."""
    numbered = caption_order(crop.lines)
    return sorted(range(len(numbered)), key=lambda i: (block_rank[numbered[i].source], numbered[i].baseline_y))


def _plan(crop: Crop, caption: Caption, text: str, block_rank: dict[str, int]) -> CropPlan:
    return CropPlan(
        rect=crop.rect,
        width=crop.size[0],
        height=crop.size[1],
        caption=text,
        lines=caption.lines,
        reading_order=_reading_order(crop, block_rank),
        rules=[crop.rule_box(rule) for rule in crop.rules],
    )


def _reference_captions(written: list[Caption], style: PageStyle, reference: Reference) -> list[str]:
    """The v2 captions ``written`` in ``style``, as a crop drawn from ``reference`` is captioned (the reference
    measured once)."""
    look, words = reference.look(), Style(provenance=style.provenance, look=style.look)
    return [
        reference_caption(
            CaptionParts(
                name=f"planned crop {k}", text=caption.text, layout=caption.layout, lines=caption.lines, style=words
            ),
            reference.caption,
            look,
        )
        for k, caption in enumerate(written)
    ]


def plan_page(layout: Layout, style: PageStyle, *, reference: Reference | None = None) -> list[CropPlan]:
    """The crops the v2 builder would cut from this page, each with its caption in ``style``, or drawn from a
    ``reference``, in the form its LoRA was trained on."""
    crops = crops_for_page(
        regions_of(layout),
        (layout.width, layout.height),
        CropSettings(),
        rules=[(rule.x0, rule.y0, rule.x1, rule.y1) for rule in layout.rules],
    )
    block_rank = {source: rank for rank, source in enumerate(dict.fromkeys(line.source for line in layout.lines))}
    written = [caption_for(crop, style.provenance, style.look) for crop in crops]
    texts = (
        [caption.text for caption in written] if reference is None else _reference_captions(written, style, reference)
    )
    return [_plan(crop, caption, text, block_rank) for crop, caption, text in zip(crops, written, texts, strict=True)]
