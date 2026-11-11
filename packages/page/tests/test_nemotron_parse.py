"""A page as Nemotron 3.5 Parse parses it: its blocks in reading order, each its box on the model's coordinate grid,
its text (a table in the LaTeX its table reader parses) and its class; its pages scaled as its processor scales them;
and, in the envs lane, that answer read back by the model's own code at its pinned revision (its postprocessing, its
tokenizer and its processor, run in ``packages/htr_train/envs/nemotron_parse`` by ``nemotron_parse_check.py``) to the
same blocks, boxes, classes and text."""

from __future__ import annotations

import html
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import diffu_htr_train.nemotron_parse as trainer
from diffu_page.cli.reader_set import app
from diffu_page.output import load_rgb
from diffu_page.page.labels import ElementLine, PageElement, PageElements
from diffu_page.page.layout import CellRef, LineKind
from diffu_page.page.layout.pages import digest
from diffu_page.page.nemotron_parse import (
    REPO,
    REVISION,
    TASK_PROMPT,
    X_BINS,
    Y_BINS,
    coordinate,
    fit_page,
    fitted_size,
    to_nemotron_parse,
)

WIDTH, HEIGHT = 1000, 2000


def _element(kind: LineKind, box: tuple[int, int, int, int], *texts: str) -> PageElement:
    """An element of ``texts``, a line each, stacked in its box."""
    x0, y0, x1, y1 = box
    step = (y1 - y0) // max(1, len(texts))
    lines = [ElementLine(text=text, box=(x0, y0 + k * step, x1, y0 + (k + 1) * step)) for k, text in enumerate(texts)]
    return PageElement(kind=kind, box=box, lines=lines)


def _table(box: tuple[int, int, int, int], cells: list[tuple[int, int, str]]) -> PageElement:
    """A table of ``cells`` (row, column, text; a text's lines split by a line break) in ``box``."""
    lines = [
        ElementLine(text=line, box=box, cell=CellRef(table="t", row=row, column=column))
        for row, column, text in cells
        for line in text.split("\n")
    ]
    return PageElement(kind=LineKind.TABLE, box=box, lines=lines)


# A page of every kind of element, in reading order: a page number at the top, a heading, running text (a blank line
# in it), a margin note, a table (a cell of two lines, one empty, LaTeX's and the table reader's specials in a cell),
# an element with nothing written, a footer, a catchword and a page number at the bottom
PAGE = PageElements(
    image="page.jpg",
    width=WIDTH,
    height=HEIGHT,
    elements=[
        _element(LineKind.PAGE_NUMBER, (900, 40, 960, 80), "12"),
        _element(LineKind.HEADING, (200, 100, 800, 200), "Förteckning öfver", "Jämtlands län"),
        _element(LineKind.BODY, (200, 300, 800, 500), "Anno 1715 den 4 maj", "  ", "kom för rätten ½ tunna\xa0råg"),
        _element(LineKind.MARGINALIA, (20, 300, 180, 400), "Nota"),
        _table(
            (200, 600, 800, 900),
            [(0, 0, "Namn"), (0, 1, "Ort & by"), (1, 0, "Anders\nPersson"), (2, 1, "50% {a_b} *c* <d> \\ ~^ $#")],
        ),
        _element(LineKind.BODY, (200, 950, 800, 1000), " "),
        _element(LineKind.FOOTER, (200, 1850, 800, 1900), "Summa 2"),
        _element(LineKind.CATCH_WORD, (700, 1920, 800, 1950), "och"),
        _element(LineKind.PAGE_NUMBER, (480, 1960, 520, 1990), "13"),
    ],
)
# Each box on the grid: x of 1000 px on 1,024 coordinates, y of 2000 px on 1,280, to four decimals (x 200 px is
# 0.2 x 1024 = 204.8, so 205 / 1024 = 0.2002; y 1960 px is 0.98 x 1280 = 1254.4, so 1254 / 1280 = 0.9797)
BLOCKS = [
    ("Page-header", (0.9004, 0.0203, 0.96, 0.0398), "12"),
    ("Section-header", (0.2002, 0.05, 0.7998, 0.1), "Förteckning öfver\nJämtlands län"),
    ("Text", (0.2002, 0.15, 0.7998, 0.25), "Anno 1715 den 4 maj\nkom för rätten 1⁄2 tunna råg"),
    ("Text", (0.0195, 0.15, 0.1797, 0.2), "Nota"),
]
TAIL = [
    ("Page-footer", (0.2002, 0.925, 0.7998, 0.95), "Summa 2"),
    ("Page-footer", (0.7002, 0.9602, 0.7998, 0.975), "och"),
    ("Page-footer", (0.4805, 0.9797, 0.5195, 0.9953), "13"),
]
TABLE_LATEX = (
    "\\begin{tabular}{cc}\n"
    "Namn & Ort \\& by\\\\\n"
    "Anders\\linebreak{}Persson & \\\\\n"
    " & 50\\% \\{a\\_b\\} \\*c\\* \\textless{}d\\textgreater{} \\textbackslash{} "
    "\\textasciitilde{}\\textasciicircum{} \\$\\#\\\\\n"
    "\\end{tabular}"
)
TABLE = ("Table", (0.2002, 0.3, 0.7998, 0.45), TABLE_LATEX)
EXPECTED = [*BLOCKS, TABLE, *TAIL]  # in reading order: the element with nothing written is left out
ANSWER = "\n\n".join(
    f"<x_{x0}><y_{y0}>{text}<x_{x1}><y_{y1}><class_{name}>" for name, (x0, y0, x1, y1), text in EXPECTED
)


def test_a_page_is_answered_as_its_blocks_in_reading_order_on_the_models_grid_text_normalized_as_it_reads() -> None:
    assert to_nemotron_parse(PAGE) == ANSWER


def test_the_sets_are_written_for_the_model_and_prompt_htr_train_trains() -> None:
    assert (REPO, REVISION, f"<image>{TASK_PROMPT}") == (trainer.REPO, trainer.REVISION, trainer.USER_TURN)


@pytest.mark.parametrize(
    ("fraction", "bins", "expected"),
    [
        pytest.param(0.0, X_BINS, "0.0", id="the first"),
        pytest.param(-0.01, X_BINS, "0.0", id="off the page: the first"),
        pytest.param(0.5, X_BINS, "0.5", id="the middle, to its shortest decimals"),
        pytest.param(3.4 / 1024, X_BINS, "0.0029", id="the nearest: 3 / 1024"),
        pytest.param(3.6 / 1024, X_BINS, "0.0039", id="the nearest: 4 / 1024"),
        pytest.param(1.0, X_BINS, "0.999", id="the right edge: the last x"),
        pytest.param(1.0, Y_BINS, "0.9992", id="the bottom edge: the last y"),
        pytest.param(1 / 1280, Y_BINS, "0.0008", id="one y step"),
    ],
)
def test_a_coordinate_is_the_nearest_of_the_models_and_an_edge_its_last(
    fraction: float, bins: int, expected: str
) -> None:
    assert coordinate(fraction, bins) == expected


@pytest.mark.parametrize(
    ("size", "fitted"),
    [
        pytest.param((2289, 3855), (1216, 2048), id="a tall page: its height to 2048"),
        pytest.param((2000, 2400), (1664, 1996), id="then its width to 1664"),
        pytest.param((3000, 1000), (1664, 554), id="a wide page: its width alone"),
        pytest.param((1000, 1200), (1000, 1200), id="a page that fits: as it is"),
    ],
)
def test_a_page_is_scaled_as_the_models_processor_scales_it_to_fit_1664_x_2048(
    size: tuple[int, int], fitted: tuple[int, int]
) -> None:
    page = Image.linear_gradient("L").resize(size)

    assert fitted_size(*size) == fitted
    assert fit_page(page).size == fitted
    assert fit_page(page).mode == "RGB"


def _text_line(line_id: str, box: tuple[int, int, int, int], text: str) -> str:
    x0, y0, x1, y1 = box
    return (
        f'<TextLine id="{line_id}"><Coords points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/>'
        f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
    )


OPENING_SIZE = (4000, 2400)  # px: two pages of 2000 x 2400, the gutter at x 2000 (the middle of the free band from
# 1900 to 2100); each page fitted to 1664 x 1996
VOLUME = Path("GT-RA_court_20260101", "Vol", "Vol")


def _opening(root: Path) -> Path:
    """An opening of the GT below ``root``, its PAGE file in its volume's ``page/`` beside its scan (red grows down it,
    green across it): a page number and a line of running text on the left, a line of running text on the right."""
    scan = root / VOLUME / "o.jpg"
    scan.parent.mkdir(parents=True)
    down = Image.linear_gradient("L").resize(OPENING_SIZE)
    across = Image.linear_gradient("L").transpose(Image.Transpose.ROTATE_90).resize(OPENING_SIZE)
    Image.merge("RGB", (down, across, Image.new("L", OPENING_SIZE))).save(scan)
    regions = (
        f'<TextRegion id="pn" type="page-number">{_text_line("pn_0", (1800, 40, 1900, 100), "12")}</TextRegion>'
        f'<TextRegion id="b">{_text_line("b_0", (400, 400, 1800, 480), "Anno 1715 den 4 maj")}</TextRegion>'
        f'<TextRegion id="s">{_text_line("s_0", (2100, 1200, 3500, 1280), "Summa 2")}</TextRegion>'
    )
    xml = root / VOLUME / "page" / "o.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
        "<Metadata><Creator>prov=READ-COOP:name=PyLaia@TranskribusPlatform</Creator></Metadata>"
        f'<Page imageFilename="o.jpg" imageWidth="{OPENING_SIZE[0]}" imageHeight="{OPENING_SIZE[1]}">{regions}</Page>'
        "</PcGts>",
        encoding="utf-8",
    )
    return xml


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_reader_set_parses_each_page_of_an_opening_as_nemotron_parse_on_the_page_its_processor_would_see(
    tmp_path: Path,
) -> None:
    xml = _opening(tmp_path / "gt")
    out = tmp_path / "set" / "np.jsonl"

    result = CliRunner().invoke(app, ["--xml", str(tmp_path / "gt"), "--model", "nemotron-parse", "--out", str(out)])

    scan_id = digest(xml.relative_to(tmp_path / "gt").as_posix())
    pages = [(tmp_path / "set" / "np_pages" / f"{scan_id}_{side}.jpg").resolve() for side in ("left", "right")]
    assert result.exit_code == 0, result.output
    # each page's own grid: x of 2000 px, y of 2400 px (the right page's from the gutter, 2000 px into the scan)
    answers = [
        "<x_0.9004><y_0.0164>12<x_0.9502><y_0.0414><class_Page-header>\n\n"
        "<x_0.2002><y_0.1664>Anno 1715 den 4 maj<x_0.9004><y_0.2><class_Text>",
        "<x_0.0498><y_0.5>Summa 2<x_0.75><y_0.5336><class_Text>",
    ]
    assert _lines(out) == [
        {
            "messages": [
                {"role": "user", "content": f"<image>{TASK_PROMPT}"},
                {"role": "assistant", "content": answer},
            ],
            "images": [str(page)],
        }
        for answer, page in zip(answers, pages, strict=True)
    ]
    scan = load_rgb(tmp_path / "gt" / VOLUME / "o.jpg")
    halves = [scan.crop((0, 0, 2000, 2400)), scan.crop((2000, 0, 4000, 2400))]
    for page, half in zip(pages, halves, strict=True):
        written = load_rgb(page)
        assert written.size == (1664, 1996)
        expected = half.resize((1664, 1996), Image.Resampling.BILINEAR)
        assert np.abs(np.asarray(written, np.int16) - np.asarray(expected, np.int16)).mean() < 2  # JPEG's loss


CHECK = Path(__file__).parent / "nemotron_parse_check.py"
ENV_PYTHON = Path(__file__).parents[2] / "htr_train" / "envs" / "nemotron_parse" / ".venv" / "bin" / "python"


def _rich_html(grid: list[list[str]]) -> str:
    """A table as the model's strict table reader writes it in HTML: a ``td`` of rowspan and colspan 1 a cell, its
    text HTML-escaped, its lines joined by ``<br>``."""
    rows = [
        "".join(
            f'<td rowspan="1" colspan="1">{"<br>".join(map(html.escape, text.split(chr(10))))}</td>' for text in row
        )
        for row in grid
    ]
    return "<table>" + "".join(f"<tr>{row}</tr>" for row in rows) + "</table>"


@pytest.mark.envs
def test_the_models_own_code_reads_a_page_back_to_its_blocks_boxes_classes_and_text(tmp_path: Path) -> None:
    if not ENV_PYTHON.is_file():
        pytest.fail(f"envs/nemotron_parse is not built: uv sync --project {ENV_PYTHON.parents[2]}")
    tall = tmp_path / "tall.jpg"
    Image.linear_gradient("L").resize((2289, 3855)).convert("RGB").save(tall)
    fitted = tmp_path / "fitted.jpg"
    fit_page(Image.open(tall)).save(fitted)
    request = {
        "repo": REPO,
        "revision": REVISION,
        "answer": to_nemotron_parse(PAGE),
        "size": [WIDTH, HEIGHT],
        "x_coordinates": [coordinate(k / X_BINS, X_BINS) for k in range(X_BINS)],
        "y_coordinates": [coordinate(k / Y_BINS, Y_BINS) for k in range(Y_BINS)],
        "images": [str(tall), str(fitted)],
    }
    (tmp_path / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

    result = subprocess.run(
        [str(ENV_PYTHON), "-I", str(CHECK), str(tmp_path / "request.json")],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )

    assert result.returncode == 0, result.stderr[-4000:]
    read = json.loads(result.stdout.splitlines()[-1])
    # every coordinate this module writes is one token of the model's, and the model has no other
    assert read["coordinates_not_tokens"] == []
    assert read["model_coordinates"] == {"x": X_BINS, "y": Y_BINS}
    # the answer is tokenized and decoded back as it is: its text is already what the tokenizer normalizes it to
    assert read["decoded"] == ANSWER
    # its postprocessing reads each block's class, box (the fractions its tokens name) and text back
    assert [(block["class"], tuple(block["bbox"]), block["text"]) for block in read["blocks"]] == EXPECTED
    # each box back in the page's pixels within half a step of the grid (1000 / 1024 and 2000 / 1280 px)
    written = [element.box for element in PAGE.elements if any(line.text.strip() for line in element.lines)]
    for block, box in zip(read["blocks"], written, strict=True):
        assert [abs(a - b) <= 0.5 * 2000 / 1280 for a, b in zip(block["bbox_pixels"], box, strict=True)] == [True] * 4
    # a text block's content is its text; the table's, its cells as HTML
    assert [block["content"] for block in read["blocks"] if block["class"] != "Table"] == [
        text for name, _, text in EXPECTED if name != "Table"
    ]
    table = next(block["content"] for block in read["blocks"] if block["class"] == "Table")
    assert table == _rich_html([["Namn", "Ort & by"], ["Anders\nPersson", ""], ["", "50% {a_b} *c* <d> \\ ~^ $#"]])
    # its processor scales the tall page to the size fit_page writes, and reads a fitted page as it is (H, W)
    assert read["content_sizes"] == [[2048, 1216], [2048, 1216]]
