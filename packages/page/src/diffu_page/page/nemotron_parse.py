"""A page as Nemotron 3.5 Parse parses it (``nvidia/Nemotron-3.5-Parse`` at :data:`REVISION`), in the form its own code
at that commit writes and reads, so a set teaches it what it already writes:

- :data:`TASK_PROMPT` is its default task (its model card): boxes, classes, Markdown text, the text in pictures.
- Its answer (:func:`to_nemotron_parse`) is the page's blocks in reading order, a blank line apart, each
  ``<x_X0><y_Y0>TEXT<x_X1><y_Y1><class_C>`` (its raw output, ``reference/parse_3_5/quick20/predictions.jsonl``); its
  postprocessing reads blocks back by that pattern (``nemotron_parse/engine/markdown.py``,
  ``extract_classes_bboxes``).
- A box's corners are fractions of the page's width and height (its ``content_normalized`` frame: of the page as its
  processor scaled it, never of the padded canvas), each written as the nearest of the coordinates its tokenizer has
  (:func:`coordinate`): 1,024 for x (``<x_0.0>`` to ``<x_0.999>``: ``k / 1024`` to four decimals) and 1,280 for y
  (``k / 1280``), so a page's right and bottom edges are the last of them.
- A block's class is one of its 15 (:data:`CLASSES`), by its element's kind (:data:`CLASS_OF`); a page number is a
  page header or a page footer by where it is, as DocLayNet, whose classes these are, places one.
- A text block's lines are each on a line of its own, as transcribed, a line with nothing written left out; a table is
  LaTeX ``tabular`` in the dialect its strict table reader parses (``engine/convert/rich_table_reader.py``):
  ``\\begin{tabular}{c...c}``, each row's cells split by `` & `` and the row ended by ``\\\\`` on a line of its own,
  each cell's text escaped as its in-cell encoder escapes it (``engine/convert/hybrid_inline.py``, ``ESC``) and its
  lines joined by ``\\linebreak{}``.
- The texts are NFKC-normalized, as its tokenizer normalizes everything it reads (``tokenizer.json``): to it ``½`` is
  ``1⁄2`` and a no-break space a space, so that is what it is taught to write.

:data:`MAX_SIZE` is its recommended largest input (W x H); its processor scales a larger page down to fit it
(:func:`fitted_size`, its ``LongestMaxSizeHW``, bilinear), never a smaller one up (:func:`fit_page`).
"""

from __future__ import annotations

import unicodedata

from PIL import Image

from diffu_page.page.labels import PageElement, PageElements
from diffu_page.page.layout.model import LineKind

REPO = "nvidia/Nemotron-3.5-Parse"
REVISION = "72ad4d71dd827135ee581502363e14748008900a"  # the commit every rule here was read at
TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
MAX_SIZE = (1664, 2048)  # W x H: its recommended largest input, its processor's final_size (H, W) read as W x H
X_BINS = 1024  # its tokenizer's x coordinates: <x_0.0> ... <x_0.999>, k / 1024 to four decimals
Y_BINS = 1280  # its y coordinates: <y_0.0> ... <y_0.9992>, k / 1280

CLASSES = (
    "Text",
    "Title",
    "Section-header",
    "List-item",
    "TOC",
    "Bibliography",
    "Footnote",
    "Page-header",
    "Page-footer",
    "Picture",
    "Formula",
    "Code",
    "Table",
    "Caption",
    "Chart",
)

# Its class for each kind of element: running text and a margin note text (its classes have no margin note), a heading
# a section header (a title is a document's own), a running header a page header, and a footer, catchword or signature
# mark, page furniture under the text, a page footer. A page number is left to where it is (:func:`block_class`).
CLASS_OF: dict[LineKind, str] = {
    LineKind.BODY: "Text",
    LineKind.MARGINALIA: "Text",
    LineKind.TABLE: "Table",
    LineKind.HEADING: "Section-header",
    LineKind.HEADER: "Page-header",
    LineKind.FOOTER: "Page-footer",
    LineKind.CATCH_WORD: "Page-footer",
    LineKind.SIGNATURE_MARK: "Page-footer",
}

# How its in-cell encoder writes a character of a table cell's text (hybrid_inline.py's ESC)
CELL_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "*": r"\*",
    "{": r"\{",
    "}": r"\}",
    "\\": r"\textbackslash{}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}
CELL_LINE_BREAK = r"\linebreak{}"  # how its encoder writes a line break in a cell (a <br>)


def coordinate(fraction: float, bins: int) -> str:
    """``fraction`` of a page's side as one of its tokenizer's coordinates on a grid of ``bins``: the nearest
    ``k / bins`` (k from 0 to ``bins - 1``, so the far edge is the last), to four decimals, as its token is named."""
    k = min(bins - 1, max(0, round(fraction * bins)))
    return str(round(k / bins, 4))


def block_class(element: PageElement, page: PageElements) -> str:
    """The class of ``element``'s block (:data:`CLASS_OF`); a page number a page header in the page's top half, a page
    footer in its bottom half."""
    if element.kind is LineKind.PAGE_NUMBER:
        middle = (element.box[1] + element.box[3]) / 2
        return "Page-header" if middle < page.height / 2 else "Page-footer"
    return CLASS_OF[element.kind]


def _normalized(text: str) -> str:
    """``text`` as its tokenizer reads it: NFKC."""
    return unicodedata.normalize("NFKC", text)


def _cell(text: str | None) -> str:
    """A table cell's text as its in-cell encoder writes it: its lines' texts escaped and joined by a line break."""
    if text is None:
        return ""
    return CELL_LINE_BREAK.join(
        "".join(CELL_ESCAPES.get(char, char) for char in _normalized(line)) for line in text.split("\n")
    )


def _tabular(element: PageElement) -> str:
    """A table as LaTeX ``tabular`` in its strict reader's dialect: a centred column each, each row ended by ``\\\\``
    on a line of its own; "" for a table with nothing written in it."""
    grid = element.grid
    if all(text is None or not text.strip() for row in grid for text in row):
        return ""
    columns = len(grid[0])
    rows = "".join(" & ".join(_cell(text) for text in row) + "\\\\\n" for row in grid)
    return f"\\begin{{tabular}}{{{'c' * columns}}}\n{rows}\\end{{tabular}}"


def _text(element: PageElement) -> str:
    """A text block's lines, each on a line of its own, a line with nothing written left out."""
    return "\n".join(_normalized(line.text) for line in element.lines if line.text.strip())


def block(element: PageElement, page: PageElements) -> str | None:
    """``element`` as its block: its box's corners on its coordinate grid (x of the page's width, y of its height), its
    text (a table as ``tabular``) and its class; None for an element with nothing written in it."""
    content = _tabular(element) if element.kind is LineKind.TABLE else _text(element)
    if not content:
        return None
    x0, y0, x1, y1 = element.box
    start = f"<x_{coordinate(x0 / page.width, X_BINS)}><y_{coordinate(y0 / page.height, Y_BINS)}>"
    end = f"<x_{coordinate(x1 / page.width, X_BINS)}><y_{coordinate(y1 / page.height, Y_BINS)}>"
    return f"{start}{content}{end}<class_{block_class(element, page)}>"


def to_nemotron_parse(page: PageElements) -> str:
    """The page as Nemotron 3.5 Parse answers its task prompt: its blocks (:func:`block`) in reading order, a blank
    line apart."""
    return "\n\n".join(text for element in page.elements if (text := block(element, page)) is not None)


def fitted_size(width: int, height: int) -> tuple[int, int]:
    """The size its processor scales a ``width`` x ``height`` page to (``_resize_with_aspect_ratio``): its height to
    :data:`MAX_SIZE`'s at most, then its width, each time the other side truncated to keep the aspect; a page that fits
    as it is."""
    max_width, max_height = MAX_SIZE
    aspect = width / height
    new_width, new_height = width, height
    if height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect)
    if new_width > max_width:
        new_width = max_width
        new_height = int(new_width / aspect)
    return new_width, new_height


def fit_page(image: Image.Image) -> Image.Image:
    """``image`` in RGB as its processor scales it (:func:`fitted_size`, bilinear), so it scales it no further."""
    image = image.convert("RGB")
    size = fitted_size(image.width, image.height)
    return image if size == image.size else image.resize(size, Image.Resampling.BILINEAR)
