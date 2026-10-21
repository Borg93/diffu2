"""The style words of a v2 layout caption: read them, swap in another archive's, or hand them to a reference image.

A caption from ``build_layout_crops.caption_for`` names its style in fixed phrases: the opening sentence (scan,
archive, years, paper), the paper row, and the ink of the opening, of every line row and of every fragment row.
``restyle`` swaps those for another style (the v2 steering test: the same text and layout drawn as another
archive). ``edit_caption`` rewrites the caption as a FLUX.3 image-edit record that takes the hand, ink and paper
from a reference image instead (the v3 reference-style LoRA).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

_OPENING = re.compile(
    r"Flat, evenly lit (?P<scan>greyscale archival scan|archival scan) of a handwritten page from "
    r"(?P<title>.+?)(?:, dated (?P<years>[^.]+))?\. The (?P<paper>[a-z -]+?) paper <paper_1>"
)


class Style(BaseModel):
    """What a caption says the crop looks like: its archive (and years) and its scan, paper and ink."""

    title: str
    years: str = ""
    greyscale: bool
    paper: str
    ink: str

    @classmethod
    def of(cls, record: dict[str, Any]) -> Style:
        """The style of a v2 crop record (``metadata.jsonl`` row): its caption's archive, its scan look.

        Raises:
            ValueError: If the caption's opening is not the v2 grammar or disagrees with the record's look.
        """
        match = _OPENING.search(record["caption"])
        if match is None:
            raise ValueError(f"no v2 opening sentence in the caption of {record['file_name']}")
        style = cls(
            title=match["title"],
            years=match["years"] or "",
            greyscale=record["greyscale"],
            paper=record["paper"],
            ink=record["ink"],
        )
        if match["paper"] != style.paper or (match["scan"] == "greyscale archival scan") != style.greyscale:
            raise ValueError(f"the caption of {record['file_name']} disagrees with its paper or scan fields")
        return style


class _Phrases(BaseModel):
    """The caption phrases that carry the style, in replacement order."""

    opening: str  # the opening sentence, up to and including "<paper_1>"
    paper_row: str
    ink_opening: str
    ink_line: str  # every line row
    ink_fragment: str  # every fragment row


def _phrases(style: Style) -> _Phrases:
    scan = "greyscale archival scan" if style.greyscale else "archival scan"
    when = f", dated {style.years}" if style.years else ""
    greyscale = ", in a greyscale scan." if style.greyscale else "."
    return _Phrases(
        opening=f"Flat, evenly lit {scan} of a handwritten page from {style.title}{when}. The {style.paper} paper <paper_1>",
        paper_row=f"Aged {style.paper} paper of a bound archival volume{greyscale}",
        ink_opening=f"cursive handwriting in {style.ink} ink",
        ink_line=f"cursive script in {style.ink} ink.",
        ink_fragment=f"handwritten line in {style.ink} ink, cut off",
    )


# The style phrases of an image-edit record: an instruction to draw a new page in <ref_image_0>'s hand (FLUX.3's
# i2i_bbox annotation, diffusers modular_pipelines/flux3/prompt_upsampler.py: "Create a ..." when the references
# are drawn into a new frame, the reference cited by its token where content is taken from it).
_FROM_REFERENCE = _Phrases(
    opening=(
        "Create a flat, evenly lit archival scan of a new handwritten page in the hand, ink and paper of "
        "<ref_image_0>. The paper <paper_1>"
    ),
    paper_row="Aged paper of a bound archival volume.",
    ink_opening="cursive handwriting",
    ink_line="cursive script in the reference's hand and ink.",
    ink_fragment="handwritten line, cut off",
)
_WHOLE = [0, 0, 1000, 1000]
_TABLE_START = ' [{"id": '


def _swap(caption: str, source: Style, target: _Phrases) -> str:
    """Replace every style phrase of ``source`` in ``caption`` with ``target``'s.

    Raises:
        ValueError: If the opening, the paper row or the opening's ink isn't in the caption exactly once.
    """
    old = _phrases(source)
    for once in ("opening", "paper_row", "ink_opening"):
        if caption.count(getattr(old, once)) != 1:
            raise ValueError(f"the caption doesn't carry its {once} {getattr(old, once)!r} exactly once")
    for field in _Phrases.model_fields:
        caption = caption.replace(getattr(old, field), getattr(target, field))
    return caption


def restyle(caption: str, source: Style, target: Style) -> str:
    """``caption`` (in ``source``'s style) naming ``target``'s archive, years, scan, paper and ink instead."""
    return _swap(caption, source, _phrases(target))


def edit_caption(caption: str, source: Style) -> str:
    """``caption`` as a FLUX.3 image-edit record: draw this page's text and layout in ``<ref_image_0>``'s hand.

    The paragraph becomes an instruction citing the reference, and the layout rows the edit table: the paper taken
    from the whole reference (``from: ref_image_0``), every line and fragment new content (``from: target``).
    """
    swapped = _swap(caption, source, _FROM_REFERENCE)
    if _TABLE_START not in swapped:
        raise ValueError("no layout table after the caption's paragraph")
    split = swapped.index(_TABLE_START)
    paragraph, rows = swapped[:split], json.loads(swapped[split + 1 :])
    edit_rows = [
        {
            "id": row["id"],
            "from": "ref_image_0",
            "src_bbox": _WHOLE,
            "tgt_bbox": row["bbox"],
            "desc": row["desc"],
        }
        if row["id"] == "paper_1"
        else {
            "id": row["id"],
            "from": "target",
            "src_bbox": None,
            "tgt_bbox": row["bbox"],
            "desc": row["desc"],
        }
        for row in rows
    ]
    return f"{paragraph} {json.dumps(edit_rows, ensure_ascii=False)}"
