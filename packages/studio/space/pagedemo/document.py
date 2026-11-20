"""The page as the demo's editor holds it, and as the page pipeline draws it.

The editor's value is a :class:`Document` as JSON: the page's shape (width over height), how many line slots it holds
from top to bottom (its line pitch: every line of a page is drawn in a slot of one pitch, as the model was trained),
and its boxes, each a rectangle ``x0, y0, x1, y1`` on 0-1000 of the page with a kind and a text. A text box is a
block of line slots stacked from its top edge; its text is written into them as the page pipeline writes a text into a
layout given as boxes (:func:`diffu_page.page.layout.refill`: a line takes about 3.7 characters per unit of its width
over its height, a word that overruns is split with ``¬``), a line break of the text starting a new line. A rule box
is a ruled line along its longer side.

:func:`page_layout` turns a document into the pipeline's :class:`~diffu_page.page.layout.Layout` at the size the page
is drawn at, each slot exactly the box the model is given for the line (its baseline 0.7 of the way down, its sides
inset by the margin the caption adds back).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from diffu_page.draw.plan import whole_page_size
from diffu_page.page.crops.geometry import SLOT_ABOVE_BASELINE, SLOT_SIDE_MARGIN
from diffu_page.page.layout import Layout, LineKind, Production, Rule, TextLine, refill

GRID = 1000  # a box's coordinates: 0 to GRID across the page, and down it
SPARE_LINES = 400  # slots laid after a box's own to catch the text it has no room for
RULE_WIDTH = 2.0  # px: a ruled line's thickness on the drawn page


class BoxKind(StrEnum):
    """What a box holds: lines of text with a place on the page, or a ruled line."""

    TEXT = "text"
    HEADING = "heading"
    NOTE = "note"
    NUMBER = "number"
    RULE = "rule"


LINE_KINDS: dict[BoxKind, LineKind] = {
    BoxKind.TEXT: LineKind.BODY,
    BoxKind.HEADING: LineKind.HEADING,
    BoxKind.NOTE: LineKind.MARGINALIA,
    BoxKind.NUMBER: LineKind.PAGE_NUMBER,
}


class Box(BaseModel):
    """One box of the page: its rectangle ``x0, y0, x1, y1`` on 0-1000 of the page, what it holds, its text and
    whether the text is printed (type, not handwriting)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=40)
    kind: BoxKind = BoxKind.TEXT
    rect: tuple[float, float, float, float]
    text: str = Field(default="", max_length=20_000)
    printed: bool = False

    @field_validator("rect")
    @classmethod
    def _on_the_page(cls, rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = rect
        if not (0 <= x0 < x1 <= GRID and 0 <= y0 < y1 <= GRID):
            raise ValueError(f"the box {rect} does not lie on the page (0 to {GRID}) with a width and a height")
        return rect


class Document(BaseModel):
    """The editor's page: its shape, its line slots from top to bottom and its boxes."""

    model_config = ConfigDict(frozen=True)

    aspect: float = Field(default=0.7, ge=0.4, le=2.5)  # width over height
    lines: int = Field(default=30, ge=8, le=60)  # line slots down the whole page: the pitch is the height over it
    boxes: list[Box] = Field(default_factory=list, max_length=80)

    @model_validator(mode="after")
    def _boxes_named_once(self) -> Self:
        names = [box.id for box in self.boxes]
        if len(set(names)) != len(names):
            raise ValueError("two boxes share a name")
        return self

    @classmethod
    def parse(cls, value: str | None) -> Document:
        """The document an editor's value holds.

        Raises:
            ValueError: If the value is not a document, naming what is wrong with it.
        """
        try:
            return cls.model_validate_json(value or "{}")
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(map(str, problem['loc'])) or 'page'}: {problem['msg']}" for problem in error.errors()
            )
            raise ValueError(f"the page cannot be read: {problems}") from error


class PageLayout(BaseModel):
    """A document as the pipeline draws it: its layout at the drawn size, and what was left out on the way (a box
    without text, words a box had no room for), in words for the page's author."""

    layout: Layout
    notes: list[str]


def page_size(aspect: float, megapixels: float) -> tuple[int, int]:
    """The size (width, height) px a page of ``aspect`` (width over height) is drawn at: about ``megapixels``, each
    side on the latent grid, as the pipeline plans a page whole."""
    return whole_page_size(round(GRID * aspect), GRID, round(megapixels * 1_000_000))


def slots(box: Box, pitch: float, size: tuple[int, int]) -> list[TextLine]:
    """The line slots of a text box on a page of ``size`` px at line pitch ``pitch`` px: as many whole pitches as the
    box is tall (one at least), stacked from its top edge and kept on the page, each a line of running text without
    a text yet."""
    width, height = size
    x0, y0, x1, y1 = box.rect
    left, right = x0 * width / GRID, x1 * width / GRID
    top, bottom = y0 * height / GRID, y1 * height / GRID
    margin = SLOT_SIDE_MARGIN * pitch
    if right - left > 6 * margin:  # the caption widens a line's box by this margin: the slot is then the box drawn
        left, right = left + margin, right - margin
    count = max(1, int((bottom - top) / pitch + 0.25))
    top = max(0.0, min(top, height - count * pitch))
    production = Production.PRINTED if box.printed else None
    return [
        TextLine(
            box=(left, top + k * pitch, right, top + (k + 1) * pitch),
            baseline_y=top + (k + SLOT_ABOVE_BASELINE) * pitch,
            block=box.id,
            production=production,
        )
        for k in range(count)
    ]


def _paragraphs(text: str) -> str:
    """``text`` with each of its lines a paragraph of its own, so a line break of the author's starts a new line."""
    return "\n\n".join(line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip())


def written(lines: list[TextLine], text: str, size: tuple[int, int]) -> tuple[list[TextLine], int]:
    """``text`` written into ``lines`` (a box's slots) as the pipeline writes a text into a layout of boxes: the
    lines that took a part of it, and how many words found no room."""
    if not lines:
        return [], len(text.split())
    spare = [lines[-1].model_copy(update={"block": "spare"}) for _ in range(SPARE_LINES)]
    page = Layout(width=size[0], height=size[1], lines=[*lines, *spare])
    filled = refill(page, _paragraphs(text)).lines
    return filled[: len(lines)], sum(len(line.text.split()) for line in filled[len(lines) :])


def _rule(box: Box, size: tuple[int, int]) -> Rule:
    """A rule box as a ruled line along its longer side, through its middle."""
    width, height = size
    x0, y0, x1, y1 = box.rect
    left, right, top, bottom = x0 * width / GRID, x1 * width / GRID, y0 * height / GRID, y1 * height / GRID
    if right - left >= bottom - top:
        middle = (top + bottom) / 2
        return Rule(x0=left, y0=middle - RULE_WIDTH / 2, x1=right, y1=middle + RULE_WIDTH / 2)
    middle = (left + right) / 2
    return Rule(x0=middle - RULE_WIDTH / 2, y0=top, x1=middle + RULE_WIDTH / 2, y1=bottom)


def page_layout(document: Document, megapixels: float) -> PageLayout:
    """``document`` as a layout at its drawn size: every text box's slots holding its text, in reading order (boxes
    from the top of the page down, then left to right), and every rule box a ruled line.

    Raises:
        ValueError: If no box of the page holds a text to draw.
    """
    size = page_size(document.aspect, megapixels)
    pitch = size[1] / document.lines
    lines: list[TextLine] = []
    rules: list[Rule] = []
    notes: list[str] = []
    for box in sorted(document.boxes, key=lambda box: (box.rect[1], box.rect[0])):
        if box.kind is BoxKind.RULE:
            rules.append(_rule(box, size))
            continue
        if not box.text.strip():
            notes.append(f"{box.id} holds no text and was left out")
            continue
        filled, left_over = written(slots(box, pitch, size), box.text, size)
        if left_over:
            notes.append(f"{box.id} had no room for its last {left_over} word{'s' if left_over != 1 else ''}")
        lines.extend(line.model_copy(update={"kind": LINE_KINDS[box.kind]}) for line in filled)
    if not lines:
        raise ValueError("no box holds a text to draw: draw a box and write its text, or load a template")
    return PageLayout(layout=Layout(width=size[0], height=size[1], lines=lines, rules=rules), notes=notes)
