"""A page layout: its size and its text lines in reading order, each with its box, baseline and region."""

from __future__ import annotations

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


class Layout(BaseModel):
    """A page's size (px) and its text lines in reading order."""

    width: int
    height: int
    boxes: list[LineBox] = Field(default_factory=list)
