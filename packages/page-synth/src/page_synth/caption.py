"""The FLUX.3 layout prompt of a crop (BFL's layout-prompt guide): a paragraph citing every line, then the layout rows."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from pydantic import BaseModel

from page_synth.geometry import Crop, Line
from page_synth.look import ScanLook
from page_synth.provenance import Provenance

NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


# PAGE region types (``type`` or Transkribus' ``structure {type:...}``) a line's desc names; other lines are
# plain "line of text".
ROLE_NAMES = {
    "marginalia": "marginal note",
    "page-number": "page number",
    "heading": "heading",
    "header": "running header",
    "catch-word": "catchword",
    "signature-mark": "signature mark",
    "footer": "footer",
}


class LayoutRow(BaseModel):
    """One element of a FLUX.3 layout prompt."""

    id: str
    bbox: list[int]
    desc: str


class LineRecord(BaseModel):
    """A crop's text line for evaluation: its layout row id, transcription, PAGE role and box."""

    id: str
    text: str
    role: str
    bbox: list[int]


class Caption(BaseModel):
    text: str  # the prompt: the caption paragraph, a space and the layout rows as JSON
    layout: list[LayoutRow]
    lines: list[LineRecord]


def caption_order(lines: list[Line]) -> list[Line]:
    """The order a caption numbers a crop's lines in (``Sv_Text_1``, ...): top to bottom, then left to right."""
    return sorted(lines, key=lambda line: (line.baseline_y, line.box[0]))


def caption_for(crop: Crop, provenance: Provenance, look: ScanLook) -> Caption:
    """The FLUX.3 layout prompt of a crop: one English paragraph citing every element as ``<id>``, a space, the
    layout rows as JSON.

    Follows the annotation rules FLUX.3 was trained on (the ``t2i_bbox`` system prompt of its prompt upsampler)
    and BFL's layout-prompt guide: medium, setting and light first; each ``<id>`` cited beside the words that
    name it, with its position in words as well as in its box; visible details (paper and ink colour) rather
    than names alone; every visible text a row quoting its words with style and alignment. A text line's id is
    its language code + ``_Text_`` + its index in reading order (``Sv_Text_1``). Lines cut by the top or bottom
    edge get a ``fragment_k`` row, so the model learns that unprompted ink appears only where a row says so.
    """
    numbered = [(f"Sv_Text_{i}", line) for i, line in enumerate(caption_order(crop.lines), start=1)]
    fragments = [
        (f"fragment_{k}", line) for k, line in enumerate(sorted(crop.fragments, key=lambda line: line.baseline_y), 1)
    ]
    column = _body_column([line for _, line in numbered])
    layout = [
        LayoutRow(id="paper_1", bbox=[0, 0, 1000, 1000], desc=_paper_desc(look)),
        *(
            LayoutRow(
                id=row_id,
                bbox=crop.slot(line),
                desc=_line_desc(_line_kind(line, column, crop.pitch), line, look),
            )
            for row_id, line in numbered
        ),
        *(
            LayoutRow(id=row_id, bbox=crop.visible_box(line), desc=_fragment_desc(crop.cut_edge(line), look))
            for row_id, line in fragments
        ),
    ]
    sentences = [
        _opening_sentence(provenance, look, numbered),
        *_role_sentences(numbered, column, crop),
        *_fragment_sentences(fragments, crop),
    ]
    rows = json.dumps([row.model_dump() for row in layout], ensure_ascii=False)
    return Caption(
        text=f"{' '.join(sentences)} {rows}",
        layout=layout,
        lines=[
            LineRecord(id=row_id, text=line.text, role=line.role, bbox=crop.slot(line)) for row_id, line in numbered
        ],
    )


def _body_column(lines: list[Line]) -> tuple[float, float]:
    """The x-range of the crop's main text (of all its lines when every line has a role)."""
    body = [line for line in lines if not line.role] or lines
    return min(line.box[0] for line in body), max(line.box[2] for line in body)


def _line_kind(line: Line, column: tuple[float, float], pitch: float) -> str:
    """How a line's desc opens: its role, or how a main-text line sits in its column."""
    if line.role:
        return ROLE_NAMES[line.role].capitalize()
    indent = (line.box[0] - column[0]) / pitch
    gap = (column[1] - line.box[2]) / pitch
    if indent > 2 and gap > 2 and abs(indent - gap) < 1.5:
        return "Centered handwritten line of text"
    if indent > 1.5:
        return "Indented handwritten line of text"
    return "Handwritten line of text"


def _line_desc(kind: str, line: Line, look: ScanLook) -> str:
    return f'{kind} reading "{line.text}", cursive script in {look.ink} ink.'


def _paper_desc(look: ScanLook) -> str:
    scan = ", in a greyscale scan." if look.greyscale else "."
    return f"Aged {look.paper} paper of a bound archival volume{scan}"


def _fragment_desc(edge: str, look: ScanLook) -> str:
    part = "lower" if edge == "top" else "upper"
    return (
        f"The {part} part of a handwritten line in {look.ink} ink, cut off by the {edge} edge of the scan; "
        "its words are not legible."
    )


def _opening_sentence(provenance: Provenance, look: ScanLook, numbered: list[tuple[str, Line]]) -> str:
    roles = Counter(line.role for _, line in numbered if line.role)
    parts = [f"a {ROLE_NAMES[r]}" if n == 1 else f"{_count(n)} {ROLE_NAMES[r]} lines" for r, n in roles.items()]
    including = f", including {' and '.join(parts)}" if parts else ""
    scan = "greyscale archival scan" if look.greyscale else "archival scan"
    when = f", dated {provenance.years}" if provenance.years else ""
    opening = (
        f"Flat, evenly lit {scan} of a handwritten page from {provenance.title}{when}. The {look.paper} paper "
        f"<paper_1> carries {len(numbered)} lines of cursive handwriting in {look.ink} ink{including}"
    )
    body = [row_id for row_id, line in numbered if not line.role]
    if not body:
        return f"{opening}."
    return f"{opening}; the main text runs from top to bottom: {_cite(body)}."


def _role_sentences(numbered: list[tuple[str, Line]], column: tuple[float, float], crop: Crop) -> list[str]:
    """One sentence per role and position: "A marginal note <Sv_Text_4> sits in the left margin."."""
    placed: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row_id, line in numbered:
        if line.role:
            placed[(line.role, _role_position(line, column, crop))].append(row_id)
    return [_role_sentence(ROLE_NAMES[role], where, ids) for (role, where), ids in placed.items()]


def _role_position(line: Line, column: tuple[float, float], crop: Crop) -> str:
    if line.role != "marginalia":
        return crop.place(line.middle_x, line.baseline_y)
    return "in the left margin" if line.middle_x < sum(column) / 2 else "in the right margin"


def _role_sentence(name: str, where: str, ids: list[str]) -> str:
    if len(ids) == 1:
        return f"A {name} {_cite(ids)} sits {where}."
    return f"{_count(len(ids)).capitalize()} {name} lines {_cite(ids)} sit {where}."


def _fragment_sentences(fragments: list[tuple[str, Line]], crop: Crop) -> list[str]:
    if not fragments:
        return []
    edges = {crop.cut_edge(line) for _, line in fragments}
    along = "the top and bottom edges" if len(edges) == 2 else f"the {edges.pop()} edge"
    return [f"Cut-off parts of further lines {_cite([row_id for row_id, _ in fragments])} show along {along}."]


def _cite(ids: list[str]) -> str:
    return " ".join(f"<{row_id}>" for row_id in ids)


def _count(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)
