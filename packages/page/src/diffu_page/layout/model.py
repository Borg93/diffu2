"""A page layout: its size, its text lines in reading order (each with its box, baseline, kind, how it was made and
its table cell) and its ruled lines."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class LineKind(StrEnum):
    """What a line is on its page, named by the PAGE type of its region: running text (the lines refill rewrites), a
    table cell's line, or a line with a role on the page (:data:`ROLE_NAMES`)."""

    BODY = "body"
    TABLE = "table"
    MARGINALIA = "marginalia"
    PAGE_NUMBER = "page-number"
    HEADING = "heading"
    HEADER = "header"
    CATCH_WORD = "catch-word"
    SIGNATURE_MARK = "signature-mark"
    FOOTER = "footer"


# The kinds that give a line a role on the page (a PAGE region ``@type`` or Transkribus' ``structure {type:...}``),
# and what a caption calls that role; a line in any other text region is running text.
ROLE_NAMES: dict[LineKind, str] = {
    LineKind.MARGINALIA: "marginal note",
    LineKind.PAGE_NUMBER: "page number",
    LineKind.HEADING: "heading",
    LineKind.HEADER: "running header",
    LineKind.CATCH_WORD: "catchword",
    LineKind.SIGNATURE_MARK: "signature mark",
    LineKind.FOOTER: "footer",
}


class Production(StrEnum):
    """How a line was made: printed (type or typewriter) or by hand, as PAGE's ``@production`` tells it."""

    PRINTED = "printed"
    HANDWRITTEN = "handwritten"


class CellRef(BaseModel):
    """The table cell a line is written in: its table (the PAGE ``TableRegion`` id), its row and its column."""

    model_config = {"frozen": True}

    table: str
    row: int
    column: int


class Rule(BaseModel):
    """A ruled line of the page (a PAGE ``SeparatorRegion``): its box, page px."""

    model_config = {"frozen": True}

    x0: float
    y0: float
    x1: float
    y1: float


class LineBox(BaseModel):
    """One text line of a page: its box and baseline height (page px), its text and where it belongs: its kind
    (running text, a table cell's line or a role, :class:`LineKind`) and its source block."""

    text: str
    x: float
    y: float
    w: float
    h: float
    baseline_y: float
    region: LineKind = LineKind.BODY
    block: str = ""  # the PAGE TextRegion / TableCell or ALTO TextBlock id the line came from
    production: Production = Production.HANDWRITTEN
    cell: CellRef | None = None  # set for a table cell's line

    @property
    def source(self) -> str:
        """The block the line came from, as reading order tracks it: its region and block."""
        return f"{self.region}|{self.block}"


class Layout(BaseModel):
    """A page's size (px), its text lines in reading order and its ruled lines."""

    width: int
    height: int
    boxes: list[LineBox] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
