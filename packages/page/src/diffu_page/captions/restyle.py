"""The style words of a v2 layout caption: swap in another archive's, or hand them to a reference image.

A caption from ``diffu_page.captions.prompt.caption_for`` is its :class:`CaptionTemplate` rendered with the phrases of
the crop's ``PageStyle``: the paragraph's opening sentence and its ink, the paper row, the ink of every handwritten
line row and of every fragment row (``phrases_for``). A printed line's row (black printer's ink whatever the scan) and
a ruled line's row hold no phrase. Each function here renders a :class:`CaptionParts`' template (a training record's,
or a planned crop's) with other phrases:

- ``restyle``: another style's archive and look (the v2 steering test: the same text and layout as another archive);
- ``edit_caption``: a FLUX.3 image-edit record that takes the hand, ink and paper from ``<ref_image_0>`` and names
  none of them (v3, v3c);
- ``named_edit_caption``: that record naming the archive and look in words as well (v3b);
- ``described_edit_caption``: that record describing the reference's measured hand, ink and paper (v3d): BFL's
  captions state a style's visible features, and FLUX.3 takes its style from the caption more than from a reference.

Training captions (from records) and generation captions (from crop plans) go through the same functions, and a
record keeps the template its own caption was rendered from, so a crop is drawn from exactly the caption grammar it
was trained on.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from diffu_page.captions.hand import ReferenceLook, hand_words, slant_words
from diffu_page.captions.prompt import (
    CaptionTemplate,
    LineRecord,
    PageStyle,
    StylePhrases,
    caption_text,
    dated,
    holds_print,
    page_words,
    phrases_for,
    scan_ending,
    scan_words,
)
from diffu_page.captions.records import CropRecord
from diffu_page.crops.geometry import LayoutBox


class ReferenceCaption(StrEnum):
    """How a caption drawn from a reference image speaks of the reference's style; it must be the form the LoRA was
    trained on."""

    PLAIN = "plain"  # only by <ref_image_0> (edit_caption: v3, v3c)
    DESCRIBED = "described"  # the reference's measured hand, ink and paper in words too (described_edit_caption: v3d)


class CaptionParts(BaseModel):
    """A v2 caption's template, its lines and the style its own caption names."""

    name: str  # what an error names it by: a crop's file name, or a planned crop's
    template: CaptionTemplate
    lines: list[LineRecord]
    style: PageStyle

    @classmethod
    def of(cls, record: CropRecord) -> CaptionParts:
        """A training record's caption.

        Raises:
            ValueError: If the record keeps no caption template (built before records kept one).
        """
        if record.template is None:
            raise ValueError(
                f"{record.file_name}: its record keeps no caption template; rebuild its crops with "
                "diffu_page.data.build_crops"
            )
        return cls(name=record.file_name, template=record.template, lines=record.lines, style=record.style)

    @property
    def printed(self) -> bool:
        """Whether the crop holds a printed line: its page is then one printed and filled in by hand."""
        return holds_print(self.lines)


def _phrases(style: PageStyle, parts: CaptionParts) -> StylePhrases:
    """The style phrases of ``parts``' caption written in ``style``."""
    return phrases_for(style, printed=parts.printed)


# An image-edit record draws a new page in <ref_image_0>'s hand: FLUX.3's i2i_bbox annotation (diffusers
# modular_pipelines/flux3/prompt_upsampler.py), "Create a ..." when the references are drawn into a new frame and
# the reference cited by its token where content is taken from it.
def _from_reference(parts: CaptionParts) -> StylePhrases:
    """The image-edit phrases that take the hand, ink and paper from the reference and name none of them."""
    return StylePhrases(
        opening=(
            f"Create a flat, evenly lit archival scan of a new {page_words(parts.printed)} in the hand, ink and paper "
            "of <ref_image_0>. The paper <paper_1>"
        ),
        ink_opening="cursive handwriting",
        paper_row="Aged paper of a bound archival volume.",
        ink_line="cursive script in the reference's hand and ink.",
        ink_fragment="handwritten line, cut off",
    )


_WHOLE_IMAGE: LayoutBox = (0, 0, 1000, 1000)


def _from_reference_naming(parts: CaptionParts) -> StylePhrases:
    """The image-edit phrases that also name the caption's archive, years, scan, paper and ink."""
    style = parts.style
    words = _phrases(style, parts)
    return StylePhrases(
        opening=(
            f"Create a flat, evenly lit {scan_words(style.look.greyscale)} of a new {page_words(parts.printed)} from "
            f"{style.provenance.title}{dated(style.provenance)}, in the hand, ink and paper of <ref_image_0>. "
            f"The {style.look.paper} paper <paper_1>"
        ),
        ink_opening=words.ink_opening,
        paper_row=words.paper_row,
        ink_line=f"cursive script in the reference's hand, in {style.look.ink} ink.",
        ink_fragment=words.ink_fragment,
    )


def _from_reference_describing(parts: CaptionParts, reference: ReferenceLook) -> StylePhrases:
    """The image-edit phrases that describe the reference's look: the hand first (BFL: what matters most leads),
    then the ink and paper by name and hex; every line row in the reference's hand and ink.

    Raises:
        ValueError: If the reference's ink or paper colour was not measured.
    """
    scan, hand = reference.scan, reference.hand
    if hand.ink is None or hand.paper is None:
        raise ValueError(f"a reference without a measured ink and paper colour ({hand}) cannot be described")
    return StylePhrases(
        opening=(
            f"Create a flat, evenly lit {scan_words(scan.greyscale)} of a new {page_words(parts.printed)} in the hand, "
            "ink and paper of <ref_image_0>: "
            f"{hand_words(hand)}, in {scan.ink} ink ({hand.ink}) on {scan.paper} paper ({hand.paper}). "
            f"The paper <paper_1>"
        ),
        ink_opening=f"cursive handwriting in {scan.ink} ink",
        paper_row=f"Aged {scan.paper} paper ({hand.paper}) of a bound archival volume{scan_ending(scan.greyscale)}",
        ink_line=f"{slant_words(hand.slant)} in the reference's hand, in {scan.ink} ink.",
        ink_fragment=f"handwritten line in {scan.ink} ink, cut off",
    )


class EditRow(BaseModel):
    """One row of a FLUX.3 image-edit layout table, in the field order of its annotation."""

    model_config = {"populate_by_name": True}

    id: str
    source: str = Field(serialization_alias="from")  # "ref_image_0" for content taken from it, else "target"
    src_bbox: LayoutBox | None
    tgt_bbox: LayoutBox
    desc: str


def restyle(parts: CaptionParts, target: PageStyle) -> str:
    """The caption naming ``target``'s archive, years, scan, paper and ink instead of its own."""
    return caption_text(*parts.template.render(_phrases(target, parts)))


def edit_caption(parts: CaptionParts) -> str:
    """The caption as a FLUX.3 image-edit record: its text and layout drawn in ``<ref_image_0>``'s hand.

    The paragraph becomes an instruction citing the reference and the layout rows the edit table: the paper taken
    from the whole reference, every line and fragment new content.
    """
    return _edit(parts, _from_reference(parts))


def named_edit_caption(parts: CaptionParts) -> str:
    """``edit_caption`` that also names the caption's archive, years, scan, paper and ink: the style in words and by
    example at once."""
    return _edit(parts, _from_reference_naming(parts))


def described_edit_caption(parts: CaptionParts, reference: ReferenceLook) -> str:
    """``edit_caption`` that describes the reference's hand, ink and paper as measured (``reference_look``)."""
    return _edit(parts, _from_reference_describing(parts, reference))


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


def _edit(parts: CaptionParts, phrases: StylePhrases) -> str:
    paragraph, rows = parts.template.render(phrases)
    edit_rows = [  # the paper taken from the whole reference, every other row new content
        EditRow(
            id=row.id,
            source="ref_image_0" if row.id == "paper_1" else "target",
            src_bbox=_WHOLE_IMAGE if row.id == "paper_1" else None,
            tgt_bbox=row.bbox,
            desc=row.desc,
        )
        for row in rows
    ]
    return caption_text(paragraph, edit_rows)
