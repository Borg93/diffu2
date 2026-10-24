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


type PageBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in page pixels


class TextLine(BaseModel):
    """One text line of a page, from the reader through the crops: its box and baseline height (page px), its text,
    its kind (running text, a table cell's line or a role, :class:`LineKind`), the block it came from, how it was made
    and, in a table, its cell."""

    model_config = {"frozen": True}

    text: str
    box: PageBox
    baseline_y: float
    kind: LineKind = LineKind.BODY
    block: str = ""  # the PAGE TextRegion / TableCell or ALTO TextBlock it came from: its id, else its place
    production: Production = Production.HANDWRITTEN
    cell: CellRef | None = None  # set for a table cell's line

    @property
    def source(self) -> str:
        """The block the line came from, as reading order tracks it: its kind and block."""
        return f"{self.kind}|{self.block}"

    @property
    def role(self) -> LineKind | None:
        """The line's role on the page (a margin note, a page number, ...); None for running text and table cells."""
        return self.kind if self.kind in ROLE_NAMES else None

    @property
    def width(self) -> float:
        """The line's width (page px)."""
        return self.box[2] - self.box[0]

    @property
    def middle_x(self) -> float:
        """The x of the line's middle."""
        return (self.box[0] + self.box[2]) / 2


class Layout(BaseModel):
    """A page's size (px), its text lines in reading order and its ruled lines."""

    width: int
    height: int
    lines: list[TextLine] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
