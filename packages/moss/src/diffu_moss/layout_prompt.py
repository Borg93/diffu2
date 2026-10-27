"""A crop's FLUX.3 layout caption as a MOSS-Image Layout prompt, from the facts the caption carries and no others: its
prose as the summary, then one item per row of its layout list: a line (a row whose id names one of the crop's lines)
as a text item with its exact text and its look, the paper, a cut-off line fragment or a ruled line as an object.

The caption is the page pipeline's (``diffu_page.captions.prompt.caption_for``): a paragraph citing every row as
``<id>``, a space, then the rows as JSON, each a box ``y0, x0, y1, x1`` on 0-1000 of the crop and a description that
quotes a line's text (``... of text reading "..."``, or ``... reading "..."`` for a printed line). A Layout prompt is
compiled as the MOSS prompt-enhancer skill's ``compile_prompt.py`` compiles an ``abs`` ``xyxy`` draft: each box's
edges floored (top, left) and ceiled (bottom, right) to whole pixels, then taken from the canvas centre
(``abs_centralized``) in ``yxyx`` order; compact JSON, non-ASCII kept.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping

from pydantic import AliasChoices, BaseModel, Field

LAYOUT_META = ["archival document scan", "handwritten manuscript"]
TOKEN = re.compile(r" ?<[A-Za-z_]+_\d+>")
LINE_LIST = "; the main text runs from top to bottom:."  # the caption's line list, once its citations are gone


class LayoutEntry(BaseModel):
    """One row of a caption's layout list: the paper, a line, a cut-off fragment or a ruled line, its box
    ``y0, x0, y1, x1`` on 0-1000 of the crop (``tgt_bbox`` in a reference-drawn caption)."""

    id: str
    bbox: list[int] = Field(validation_alias=AliasChoices("bbox", "tgt_bbox"))
    desc: str


def caption_layout(caption: str) -> list[LayoutEntry]:
    """The layout list a caption ends with (its JSON rows after the prose).

    Raises:
        ValueError: If the caption has no layout list.
    """
    start = caption.find(" [{")
    if start < 0:
        raise ValueError(f"no layout list in caption: {caption[:120]!r}")
    return [LayoutEntry.model_validate(row) for row in json.loads(caption[start + 1 :])]


def summary_caption(caption: str) -> str:
    """The caption's prose, without its row citations, its line list and its JSON rows."""
    return TOKEN.sub("", caption.split(" [{", 1)[0]).replace(LINE_LIST, ".")


def centred_box(bbox: list[int], width: int, height: int) -> list[int]:
    """A 0-1000 ``y0, x0, y1, x1`` box of a ``width`` x ``height`` canvas as whole pixels from its centre, ``yxyx``:
    the top and left edges floored, the bottom and right ceiled."""
    y0, x0, y1, x1 = (edge * side / 1000 for edge, side in zip(bbox, (height, width, height, width), strict=True))
    return [
        math.floor(round(y0, 9)) - height // 2,
        math.floor(round(x0, 9)) - width // 2,
        math.ceil(round(y1, 9)) - height // 2,
        math.ceil(round(x1, 9)) - width // 2,
    ]


def appearance(desc: str, text: str) -> str:
    """A line's description without the words it quotes (``of text reading "..."`` or ``reading "..."``).

    Raises:
        ValueError: If the description does not quote the text.
    """
    for quoted in (f' of text reading "{text}"', f' reading "{text}"'):
        if quoted in desc:
            return desc.replace(quoted, "", 1)
    raise ValueError(f"the description {desc[:80]!r} does not quote its text {text!r}")


def layout_item(entry: LayoutEntry, texts: Mapping[str, str], size: tuple[int, int]) -> dict[str, object]:
    """A layout row as a MOSS item: a line (its id in ``texts``) as a text item, its look as the caption and its text
    verbatim; the paper, a fragment or a rule as an object."""
    box = centred_box(entry.bbox, *size)
    if entry.id not in texts:
        return {"label": "object", "caption": entry.desc, "bbox": box}
    text = texts[entry.id]
    return {"label": "text", "caption": appearance(entry.desc, text), "text": text, "bbox": box}


def layout_prompt(caption: str, texts: Mapping[str, str], size: tuple[int, int]) -> str:
    """The crop captioned ``caption`` as a compiled MOSS Layout prompt: the summary, then an item per row, a text item
    for each line of ``texts`` (each line's text by its row id), on a canvas of ``size`` (width, height) px.

    Raises:
        ValueError: If the caption has no layout list, or a line's row does not quote its text.
    """
    items = [layout_item(entry, texts, size) for entry in caption_layout(caption)]
    summary = {"meta": LAYOUT_META, "canvas": "single", "caption": summary_caption(caption)}
    prompt = {"summary": summary, "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(prompt, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
