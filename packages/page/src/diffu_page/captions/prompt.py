"""The FLUX.3 layout prompt of a crop (BFL's layout-prompt guide): a paragraph citing every line, then the layout rows."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict

from diffu_page.captions.look import ScanLook
from diffu_page.captions.provenance import Provenance
from diffu_page.crops.geometry import Box, Crop, Line
from diffu_page.layout.model import ROLE_NAMES, CellRef, Production

NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
HANDWRITTEN_PAGE = "handwritten page"
PRINTED_PAGE = "page printed and filled in by hand"  # a crop that holds printed lines


class LayoutRow(BaseModel):
    """One element of a FLUX.3 layout prompt."""

    id: str
    bbox: list[int]
    desc: str


class LineRecord(BaseModel):
    """A crop's text line: its layout row id, transcription, PAGE role, box, baseline, the template block it came
    from, how it was made and, in a table, its cell."""

    id: str
    text: str
    role: str
    bbox: list[int]
    # 0-1000 of the crop height. None in records written before baselines were kept: their labels put it 0.7 of the
    # way down the box, which is wrong where a crop edge cuts the box.
    baseline: int | None = None
    block: str = ""  # "" in records written before blocks were kept: their labels group lines by role alone
    production: Production = Production.HANDWRITTEN
    cell: CellRef | None = None


class Caption(BaseModel):
    """A crop's FLUX.3 layout prompt with the layout rows and line records it was written from."""

    text: str  # the prompt: the caption paragraph, a space and the layout rows as JSON
    layout: list[LayoutRow]
    lines: list[LineRecord]


class StylePhrases(BaseModel):
    """The phrases of a caption that carry its style (the archive, its years, the scan, the paper and the ink): the
    one place the caption grammar spells them, so restyling (:mod:`diffu_page.captions.restyle`) rewrites exactly
    what :func:`caption_for` wrote."""

    model_config = ConfigDict(frozen=True)

    opening: str  # the paragraph's first sentence, up to and including "<paper_1>"
    ink_opening: str  # the handwriting's ink in the paragraph
    paper_row: str  # the paper row's whole desc
    ink_line: str  # how every handwritten line row's desc ends
    ink_fragment: str  # inside every fragment row's desc


def phrases_for(provenance: Provenance, look: ScanLook, *, printed: bool) -> StylePhrases:
    """The style phrases of a crop's caption in ``provenance`` and ``look``; ``printed``: the crop holds printed
    lines, so its page is printed and filled in by hand."""
    scan = "greyscale archival scan" if look.greyscale else "archival scan"
    when = f", dated {provenance.years}" if provenance.years else ""
    page = PRINTED_PAGE if printed else HANDWRITTEN_PAGE
    greyscale = ", in a greyscale scan." if look.greyscale else "."
    return StylePhrases(
        opening=f"Flat, evenly lit {scan} of a {page} from {provenance.title}{when}. The {look.paper} paper <paper_1>",
        ink_opening=f"cursive handwriting in {look.ink} ink",
        paper_row=f"Aged {look.paper} paper of a bound archival volume{greyscale}",
        ink_line=f"cursive script in {look.ink} ink.",
        ink_fragment=f"handwritten line in {look.ink} ink, cut off",
    )


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
    A printed line's row says it is printed, a table cell's line's that it sits in a table cell, and each ruled
    line of the crop gets a ``rule_k`` row; a crop with none of them is captioned exactly as before.
    """
    numbered = [(f"Sv_Text_{i}", line) for i, line in enumerate(caption_order(crop.lines), start=1)]
    fragments = [
        (f"fragment_{k}", line) for k, line in enumerate(sorted(crop.fragments, key=lambda line: line.baseline_y), 1)
    ]
    rules = [(f"rule_{k}", rule) for k, rule in enumerate(crop.rules, 1)]
    column = _body_column([line for _, line in numbered if line.cell is None] or [line for _, line in numbered])
    phrases = phrases_for(provenance, look, printed=any(line.production is Production.PRINTED for _, line in numbered))
    layout = [
        LayoutRow(id="paper_1", bbox=[0, 0, 1000, 1000], desc=phrases.paper_row),
        *(
            LayoutRow(id=row_id, bbox=crop.slot(line), desc=_row_desc(line, column, crop, phrases))
            for row_id, line in numbered
        ),
        *(
            LayoutRow(id=row_id, bbox=crop.visible_box(line), desc=_fragment_desc(crop.cut_edge(line), phrases))
            for row_id, line in fragments
        ),
        *(LayoutRow(id=row_id, bbox=crop.rule_box(rule), desc=_rule_desc(rule)) for row_id, rule in rules),
    ]
    sentences = [
        _opening_sentence(phrases, numbered),
        *_role_sentences(numbered, column, crop),
        *_table_sentences(numbered),
        *_rule_sentences(rules, numbered),
        *_fragment_sentences(fragments, crop),
    ]
    rows = json.dumps([row.model_dump() for row in layout], ensure_ascii=False)
    return Caption(
        text=f"{' '.join(sentences)} {rows}",
        layout=layout,
        lines=[
            LineRecord(
                id=row_id,
                text=line.text,
                role=line.role,
                bbox=crop.slot(line),
                baseline=crop.baseline(line),
                block=line.source,
                production=line.production,
                cell=line.cell,
            )
            for row_id, line in numbered
        ],
    )


def _body_column(lines: list[Line]) -> tuple[float, float]:
    """The x-range of the crop's main text (of all its lines when every line has a role)."""
    body = [line for line in lines if not line.role] or lines
    return min(line.box[0] for line in body), max(line.box[2] for line in body)


def _row_desc(line: Line, column: tuple[float, float], crop: Crop, phrases: StylePhrases) -> str:
    """A line's layout row: printed, in a table cell, or placed in the crop's main text."""
    if line.production is Production.PRINTED:
        what = ROLE_NAMES[line.role] if line.role else "text"
        where = " in a table cell" if line.cell is not None else ""
        return f'Printed {what} reading "{line.text}"{where}, in a serif typeface in black printer\'s ink.'
    if line.cell is not None:
        return f'Handwritten line of text reading "{line.text}" in a table cell, {phrases.ink_line}'
    return _line_desc(_line_kind(line, column, crop.pitch), line, phrases)


def _rule_desc(rule: Box) -> str:
    """A ruled line's row: its direction only, as PAGE does not say whether a rule was printed or drawn."""
    direction = "vertical" if rule[3] - rule[1] > rule[2] - rule[0] else "horizontal"
    return f"A straight {direction} ruled line."


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


def _line_desc(kind: str, line: Line, phrases: StylePhrases) -> str:
    return f'{kind} reading "{line.text}", {phrases.ink_line}'


def _fragment_desc(edge: str, phrases: StylePhrases) -> str:
    part = "lower" if edge == "top" else "upper"
    return f"The {part} part of a {phrases.ink_fragment} by the {edge} edge of the scan; its words are not legible."


def _opening_sentence(phrases: StylePhrases, numbered: list[tuple[str, Line]]) -> str:
    """The medium, the page and its lines: the handwritten ones counted with their roles, the printed ones cited,
    and the main text (handwritten lines outside a role and a table) in reading order."""
    written = [(row_id, line) for row_id, line in numbered if line.production is Production.HANDWRITTEN]
    printed = [row_id for row_id, line in numbered if line.production is Production.PRINTED]
    roles = Counter(line.role for _, line in written if line.role)
    parts = [f"a {ROLE_NAMES[r]}" if n == 1 else f"{_count(n)} {ROLE_NAMES[r]} lines" for r, n in roles.items()]
    including = f", including {' and '.join(parts)}" if parts else ""
    opening = f"{phrases.opening} carries {len(written)} lines of {phrases.ink_opening}{including}"
    if printed:
        opening += f" and {len(printed)} printed lines in black printer's ink {_cite(printed)}"
    body = [row_id for row_id, line in written if not line.role and line.cell is None]
    if not body:
        return f"{opening}."
    return f"{opening}; the main text runs from top to bottom: {_cite(body)}."


def _table_sentences(numbered: list[tuple[str, Line]]) -> list[str]:
    """One sentence for the crop's table cells, when it has any: how many rows and columns they fill."""
    cells = [(row_id, line.cell) for row_id, line in numbered if line.cell is not None]
    if not cells:
        return []
    rows, columns = len({cell.row for _, cell in cells}), len({cell.column for _, cell in cells})
    return [
        f"A table of {_counted(rows, 'row')} and {_counted(columns, 'column')} holds "
        f"{_cite([row_id for row_id, _ in cells])}, row by row."
    ]


def _rule_sentences(rules: list[tuple[str, Box]], numbered: list[tuple[str, Line]]) -> list[str]:
    """One sentence for the crop's ruled lines, when it has any."""
    if not rules:
        return []
    where = "the table" if any(line.cell is not None for _, line in numbered) else "the page"
    lines = "ruled line crosses" if len(rules) == 1 else "ruled lines cross"
    return [f"{_count(len(rules)).capitalize()} straight {lines} {where}: {_cite([row_id for row_id, _ in rules])}."]


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


def _counted(n: int, noun: str) -> str:
    """``n`` in words with ``noun`` in the number it takes: "one row", "three columns"."""
    return f"{_count(n)} {noun}" if n == 1 else f"{_count(n)} {noun}s"


def _count(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)
