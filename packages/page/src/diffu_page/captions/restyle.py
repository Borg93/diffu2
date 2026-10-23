"""The style words of a v2 layout caption: swap in another archive's, or hand them to a reference image.

A caption from ``diffu_page.captions.prompt.caption_for`` is a paragraph, a space and the crop's layout rows as JSON.
Its style comes from the crop's ``Provenance`` (archive, years) and ``ScanLook`` (scan, paper, ink), in fixed phrases:
the paragraph's opening sentence and its ink, the paper row, the ink of every line row and of every fragment row. Each
function here rewrites those phrases of a :class:`CaptionParts` (a training record's caption, or a planned crop's):

- ``restyle``: another style's archive and look (the v2 steering test: the same text and layout as another archive);
- ``edit_caption``: a FLUX.3 image-edit record that takes the hand, ink and paper from ``<ref_image_0>`` and names
  none of them (v3, v3c);
- ``named_edit_caption``: that record naming the archive and look in words as well (v3b);
- ``described_edit_caption``: that record describing the reference's measured hand, ink and paper (v3d): BFL's
  captions state a style's visible features, and FLUX.3 takes its style from the caption more than from a reference.

Training captions (from records) and generation captions (from crop plans) go through the same functions, so a crop
is drawn from exactly the caption grammar it was trained on. Every phrase is built from a style's fields and must
occur where the grammar writes it, so a change to the grammar fails loudly instead of leaving stale words behind.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from enum import StrEnum

from pydantic import BaseModel, Field

from diffu_page.captions.hand import ReferenceLook, hand_words, slant_words
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LayoutRow
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.records import CropRecord


class StaleCaptionError(ValueError):
    """A record's caption lacks the style words the caption grammar writes for it now: it was built by an older
    grammar or provenance rule, and its crops need rebuilding (``diffu_page.data.build_crops``)."""


class ReferenceCaption(StrEnum):
    """How a caption drawn from a reference image speaks of the reference's style; it must be the form the LoRA was
    trained on."""

    PLAIN = "plain"  # only by <ref_image_0> (edit_caption: v3, v3c)
    DESCRIBED = "described"  # the reference's measured hand, ink and paper in words too (described_edit_caption: v3d)


class Style(BaseModel):
    """What a caption says a crop looks like: its archive (and years) and its scan, paper and ink."""

    provenance: Provenance
    look: ScanLook

    @classmethod
    def of(cls, record: CropRecord) -> Style:
        """The style a crop record's caption names."""
        return cls(provenance=record.provenance, look=record.look)


class CaptionParts(BaseModel):
    """A v2 caption and what it was written from: its layout rows and the style it names."""

    name: str  # what an error names it by: a crop's file name, or a planned crop's
    text: str
    layout: list[LayoutRow]
    style: Style

    @classmethod
    def of(cls, record: CropRecord) -> CaptionParts:
        """A training record's caption."""
        return cls(name=record.file_name, text=record.caption, layout=record.layout, style=Style.of(record))


@contextmanager
def _written_by_this_grammar(parts: CaptionParts) -> Generator[None]:
    try:
        yield
    except ValueError as error:
        raise StaleCaptionError(
            f"{parts.name}: its caption was not written by the current caption grammar ({error}); "
            "rebuild the crops with diffu_page.data.build_crops"
        ) from error


class _Phrases(BaseModel):
    """The caption phrases that carry a style."""

    opening: str  # the paragraph's first sentence, up to and including "<paper_1>"
    ink_opening: str  # the paragraph's ink
    paper_row: str  # the paper row's whole desc
    ink_line: str  # how every line row's desc ends
    ink_fragment: str  # inside every fragment row's desc


def _phrases(style: Style) -> _Phrases:
    scan = "greyscale archival scan" if style.look.greyscale else "archival scan"
    when = f", dated {style.provenance.years}" if style.provenance.years else ""
    greyscale = ", in a greyscale scan." if style.look.greyscale else "."
    return _Phrases(
        opening=(
            f"Flat, evenly lit {scan} of a handwritten page from {style.provenance.title}{when}. "
            f"The {style.look.paper} paper <paper_1>"
        ),
        ink_opening=f"cursive handwriting in {style.look.ink} ink",
        paper_row=f"Aged {style.look.paper} paper of a bound archival volume{greyscale}",
        ink_line=f"cursive script in {style.look.ink} ink.",
        ink_fragment=f"handwritten line in {style.look.ink} ink, cut off",
    )


# An image-edit record draws a new page in <ref_image_0>'s hand: FLUX.3's i2i_bbox annotation (diffusers
# modular_pipelines/flux3/prompt_upsampler.py), "Create a ..." when the references are drawn into a new frame and
# the reference cited by its token where content is taken from it.
_FROM_REFERENCE = _Phrases(
    opening=(
        "Create a flat, evenly lit archival scan of a new handwritten page in the hand, ink and paper of "
        "<ref_image_0>. The paper <paper_1>"
    ),
    ink_opening="cursive handwriting",
    paper_row="Aged paper of a bound archival volume.",
    ink_line="cursive script in the reference's hand and ink.",
    ink_fragment="handwritten line, cut off",
)
_WHOLE_IMAGE = [0, 0, 1000, 1000]


def _from_reference_naming(style: Style) -> _Phrases:
    """The image-edit phrases that also name ``style``'s archive, years, scan, paper and ink."""
    words = _phrases(style)
    scan = "greyscale archival scan" if style.look.greyscale else "archival scan"
    when = f", dated {style.provenance.years}" if style.provenance.years else ""
    return _Phrases(
        opening=(
            f"Create a flat, evenly lit {scan} of a new handwritten page from {style.provenance.title}{when}, in the "
            f"hand, ink and paper of <ref_image_0>. The {style.look.paper} paper <paper_1>"
        ),
        ink_opening=words.ink_opening,
        paper_row=words.paper_row,
        ink_line=f"cursive script in the reference's hand, in {style.look.ink} ink.",
        ink_fragment=words.ink_fragment,
    )


def _from_reference_describing(reference: ReferenceLook) -> _Phrases:
    """The image-edit phrases that describe the reference's look: the hand first (BFL: what matters most leads),
    then the ink and paper by name and hex; every line row in the reference's hand and ink."""
    scan, hand = reference.scan, reference.hand
    kind = "greyscale archival scan" if scan.greyscale else "archival scan"
    greyscale = ", in a greyscale scan." if scan.greyscale else "."
    return _Phrases(
        opening=(
            f"Create a flat, evenly lit {kind} of a new handwritten page in the hand, ink and paper of <ref_image_0>: "
            f"{hand_words(hand)}, in {scan.ink} ink ({hand.ink}) on {scan.paper} paper ({hand.paper}). "
            f"The paper <paper_1>"
        ),
        ink_opening=f"cursive handwriting in {scan.ink} ink",
        paper_row=f"Aged {scan.paper} paper ({hand.paper}) of a bound archival volume{greyscale}",
        ink_line=f"{slant_words(hand.slant)} in the reference's hand, in {scan.ink} ink.",
        ink_fragment=f"handwritten line in {scan.ink} ink, cut off",
    )


class EditRow(BaseModel):
    """One row of a FLUX.3 image-edit layout table, in the field order of its annotation."""

    model_config = {"populate_by_name": True}

    id: str
    source: str = Field(serialization_alias="from")  # "ref_image_0" for content taken from it, else "target"
    src_bbox: list[int] | None
    tgt_bbox: list[int]
    desc: str


def _rows_json(rows: list[dict[str, object]]) -> str:
    return json.dumps(rows, ensure_ascii=False)


def _paragraph(parts: CaptionParts) -> str:
    """The caption's paragraph: everything before the layout rows ``caption_for`` appended as JSON.

    Raises:
        ValueError: If the caption doesn't end in its own layout rows.
    """
    rows = f" {_rows_json([row.model_dump() for row in parts.layout])}"
    if not parts.text.endswith(rows):
        raise ValueError(f"the caption of {parts.name} doesn't end in its layout rows")
    return parts.text.removesuffix(rows)


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected {old!r} once in {text[:80]!r}...")
    return text.replace(old, new)


def _restyled_paragraph(paragraph: str, old: _Phrases, new: _Phrases) -> str:
    return _replace_once(_replace_once(paragraph, old.opening, new.opening), old.ink_opening, new.ink_opening)


def _restyled_desc(row: LayoutRow, old: _Phrases, new: _Phrases) -> str:
    """A layout row's desc in the new style: the paper row whole, a line row's ending, a fragment row's ink.

    Raises:
        ValueError: If the desc doesn't carry the phrase the grammar gives its kind of row.
    """
    if row.id == "paper_1":
        if row.desc != old.paper_row:
            raise ValueError(f"paper row {row.desc!r} is not {old.paper_row!r}")
        return new.paper_row
    if row.id.startswith("fragment_"):
        return _replace_once(row.desc, old.ink_fragment, new.ink_fragment)
    if not row.desc.endswith(old.ink_line):
        raise ValueError(f"line row {row.id} doesn't end in {old.ink_line!r}")
    return row.desc.removesuffix(old.ink_line) + new.ink_line


def restyle(parts: CaptionParts, target: Style) -> str:
    """The caption naming ``target``'s archive, years, scan, paper and ink instead of its own."""
    old, new = _phrases(parts.style), _phrases(target)
    with _written_by_this_grammar(parts):
        rows = [row.model_copy(update={"desc": _restyled_desc(row, old, new)}) for row in parts.layout]
        paragraph = _restyled_paragraph(_paragraph(parts), old, new)
    return f"{paragraph} {_rows_json([row.model_dump() for row in rows])}"


def edit_caption(parts: CaptionParts) -> str:
    """The caption as a FLUX.3 image-edit record: its text and layout drawn in ``<ref_image_0>``'s hand.

    The paragraph becomes an instruction citing the reference and the layout rows the edit table: the paper taken
    from the whole reference, every line and fragment new content.
    """
    return _edit(parts, _FROM_REFERENCE)


def named_edit_caption(parts: CaptionParts) -> str:
    """``edit_caption`` that also names the caption's archive, years, scan, paper and ink: the style in words and by
    example at once."""
    return _edit(parts, _from_reference_naming(parts.style))


def described_edit_caption(parts: CaptionParts, reference: ReferenceLook) -> str:
    """``edit_caption`` that describes the reference's hand, ink and paper as measured (``reference_look``)."""
    return _edit(parts, _from_reference_describing(reference))


def reference_caption(parts: CaptionParts, form: ReferenceCaption, reference: ReferenceLook | None) -> str:
    """The caption drawn from a reference image in ``form``, the form the LoRA in use was trained on: training pairs
    (``diffu_page.data.style_pairs``) and planned crops (``diffu_page.plan``) are both captioned here.

    Raises:
        ValueError: If ``form`` describes the reference and its measured look (``reference``) is missing.
    """
    match form:
        case ReferenceCaption.PLAIN:
            return edit_caption(parts)
        case ReferenceCaption.DESCRIBED:
            if reference is None:
                raise ValueError(f"{parts.name}: a described caption needs the reference's measured look")
            return described_edit_caption(parts, reference)


def _edit(parts: CaptionParts, new: _Phrases) -> str:
    old = _phrases(parts.style)
    with _written_by_this_grammar(parts):
        rows = [
            EditRow(
                id=row.id,
                source="ref_image_0" if row.id == "paper_1" else "target",
                src_bbox=_WHOLE_IMAGE if row.id == "paper_1" else None,
                tgt_bbox=row.bbox,
                desc=_restyled_desc(row, old, new),
            )
            for row in parts.layout
        ]
        paragraph = _restyled_paragraph(_paragraph(parts), old, new)
    return f"{paragraph} {_rows_json([row.model_dump(by_alias=True) for row in rows])}"
