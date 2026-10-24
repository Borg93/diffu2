"""A page layout: its size, its text lines in reading order (each with its box, baseline, region, how it was made and
its table cell) and its ruled lines."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

RUNNING_TEXT = "body"  # the region of running text: the lines refill rewrites
TABLE = "table"  # the region of a table cell's lines

# The PAGE region types (``@type`` or Transkribus' ``structure {type:...}``) that give a line a role on the page, and
# what a caption calls that role; a line in any other text region is running text.
ROLE_NAMES = {
    "marginalia": "marginal note",
    "page-number": "page number",
    "heading": "heading",
    "header": "running header",
    "catch-word": "catchword",
    "signature-mark": "signature mark",
    "footer": "footer",
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
    """One text line of a page: its box and baseline height (page px), its text and where it belongs.

    ``region`` is the PAGE type of the line's region when that type names a role (``marginalia``,
    ``page-number``, ``heading``, ... see :data:`ROLE_NAMES`), ``table`` for a table
    cell's line, else ``body``.
    """

    text: str
    x: float
    y: float
    w: float
    h: float
    baseline_y: float
    region: str = RUNNING_TEXT
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
