"""A document and its sample page as what the page pipeline draws: the page planned whole and captioned from its
sample, as the model was trained (the ``sample`` caption form), the sample's line boxes when its writing scale could be
measured, and the Layout prompt the model reads. CPU only: everything here runs before a GPU is asked for.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from diffu_moss.sample_caption import sample_layout_prompt
from diffu_page.captions.look import Ink, Paper, ScanLook
from diffu_page.captions.prompt import PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.draw.plan import CropPlan, Reference, plan_whole_page
from diffu_page.draw.sample import line_pitch as page_line_pitch
from diffu_page.page.crops.geometry import LayoutBox
from pagedemo.document import Document, page_layout
from pagedemo.sample import line_pitch, sample_lines

# The sample form's caption names no archive, paper or ink (its look is the sample's) and reads no file of the
# reference's, so neither this style nor this path is ever written or opened
STYLE = PageStyle(
    provenance=Provenance(title="an archive", years=""),
    look=ScanLook(greyscale=False, paper=Paper.CREAM, ink=Ink.DARK_BROWN),
)
SAMPLE = Reference(image=Path("sample"), caption=ReferenceCaption.SAMPLE)


class PagePlan(BaseModel):
    """A page ready to draw: its plan, the Layout prompt compiled from it, the sample's line boxes (none when its
    writing scale could not be measured: the sample is then sized as a whole page), and notes for the page's author."""

    plan: CropPlan
    prompt: str
    sample_lines: list[LayoutBox]
    notes: list[str]


def layout_prompt_of(plan: CropPlan) -> str:
    """The Layout prompt the model reads for ``plan`` drawn with a sample image, as the pipeline compiles it."""
    texts = {line.id: line.text for line in plan.lines}
    return sample_layout_prompt(plan.caption, texts, (plan.width, plan.height), sample=True)


def page_plan(document: Document, *, megapixels: float) -> tuple[CropPlan, list[str]]:
    """``document`` planned as one page of about ``megapixels``, captioned as a page drawn from a sample image, and
    what its layout left out, in words.

    Raises:
        ValueError: If the page holds no text to draw.
    """
    page = page_layout(document, megapixels)
    pixels = page.layout.width * page.layout.height
    return plan_whole_page(page.layout, STYLE, pixels=pixels, reference=SAMPLE), list(page.notes)


def filling_zoom(sample: Image.Image, pitch: float, plan: CropPlan) -> float:
    """How much larger than the page's line pitch a sample of line pitch ``pitch`` px is shown so that it is not
    shrunk to a piece smaller than the page: 1 when its writing is no larger than the page's; else up to its own
    size, or the size at which it covers the page if that is smaller. A sample shrunk to the page's pitch and left
    covering half the page lost its ink (``docs/MOSS_RUNS.md``, 20 Nov 2026: 28.7 from the sample's, 12.4 unshrunk)."""
    page_pitch = page_line_pitch([line.bbox for line in plan.lines], plan.height)
    matched = page_pitch / pitch  # the scale that makes the sample's writing the page's size
    covering = max(plan.width / sample.width, plan.height / sample.height)
    return max(1.0, min(1.0, covering) / matched)


def planned(document: Document, sample: Image.Image, *, megapixels: float, zoom: float) -> PagePlan:
    """``document`` planned as one page of about ``megapixels`` drawn from ``sample``, the sample shown at the page's
    line pitch times ``zoom`` when its own pitch can be measured.

    Raises:
        ValueError: If the page holds no text to draw.
    """
    plan, notes = page_plan(document, megapixels=megapixels)
    pitch = line_pitch(sample)
    lines: list[LayoutBox] = []
    if pitch is None:
        notes.append("the sample's line height could not be measured: it is shown to the model as a whole page")
    else:
        lines = sample_lines(sample, pitch, zoom=zoom * filling_zoom(sample, pitch, plan))
        plan = plan.model_copy(update={"whole_page": False})  # the sample cut at the page's line pitch, as a window's
    return PagePlan(plan=plan, prompt=layout_prompt_of(plan), sample_lines=lines, notes=notes)
