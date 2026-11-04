"""A page reader's fine-tuning set from a run's images and their elements files: a sample per page in DeepSeek-OCR 2's
conversation and in PaddleOCR-VL's (the page as Markdown, or its lines spotted), or a sample per element (its crop,
recognition prompt and text) for PaddleOCR-VL's recogniser; and from PAGE or ALTO files with their scans, the GT's or
a drawn run's: a page parsed as Unlimited-OCR 2 parses it (an opening's two pages each cut from it, or shown together)
or read as MiniCPM-OCR reads it (whole, element by element, or line by line, a crop past a pixel budget scaled to it),
split by the GT's own test lists, less the pages a list leaves out, of the collections asked for (one it does not hold
refused), its scans read in the order a seed shuffles them, with each sample's meta and each file, scan, page or line
left out recorded."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from diffu_page.cli.reader_set import app
from diffu_page.output import load_rgb
from diffu_page.page.labels import (
    ElementLine,
    LineLabel,
    PageElement,
    PageElements,
    PageLabels,
    Rect,
    to_alto,
    to_elements_json,
    to_page_xml,
)
from diffu_page.page.layout import CellRef, LineKind, PageBox, Production
from diffu_page.page.layout.pages import digest

IMAGE = "000_page_0.jpg"
HEADING = PageElement(
    kind=LineKind.HEADING,
    box=(100, 20, 700, 100),
    lines=[
        ElementLine(text="Förteckning öfver", box=(100, 20, 700, 60), production=Production.PRINTED),
        ElementLine(text="Jämtlands län", box=(100, 60, 700, 100), production=Production.PRINTED),
    ],
)
TABLE = PageElement(
    kind=LineKind.TABLE,
    box=(100, 120, 500, 200),
    lines=[
        ElementLine(text=text, box=(x, y, x + 200, y + 40), cell=CellRef(table="register", row=row, column=column))
        for text, row, column, x, y in (
            ("Namn", 0, 0, 100, 120),
            ("Ort", 0, 1, 300, 120),
            ("Anders", 1, 0, 100, 160),
            ("Bro", 1, 1, 300, 160),
        )
    ],
)
BODY = PageElement(
    kind=LineKind.BODY,
    box=(100, 300, 700, 380),
    lines=[
        ElementLine(text="Anno 1715 den 4 maj", box=(100, 300, 700, 340)),
        ElementLine(text="kom för rätten", box=(100, 340, 700, 380)),
    ],
)
LINE = PageElement(  # a page of one line of running text
    kind=LineKind.BODY,
    box=(100, 300, 700, 340),
    lines=[ElementLine(text="Anno 1715 den 4 maj", box=(100, 300, 700, 340))],
)


def _run(run: Path, *elements: PageElement) -> Path:
    """A run of one 800 x 1000 px image (shaded down the page, so crops at different heights differ) and its elements
    file."""
    run.mkdir(parents=True)
    Image.linear_gradient("L").resize((800, 1000)).convert("RGB").save(run / IMAGE)
    page = PageElements(image=IMAGE, width=800, height=1000, elements=list(elements))
    (run / "000_page_0.elements.json").write_text(f"{to_elements_json(page)}\n", encoding="utf-8")
    return run


def _lines(path: Path) -> list[dict[str, Any]]:
    """A JSON-lines file's objects: a set's samples, their meta or what it left out."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


NO_CROP = {"kind": None, "production": None, "element": None, "line": None}  # a page sample's meta: of no crop


@pytest.mark.parametrize(
    ("task", "prompt", "answer"),
    [
        pytest.param(
            [],
            "<image>\n<|grounding|>Convert the document to markdown.",
            "<|ref|>text<|/ref|><|det|>[[125, 300, 874, 340]]<|/det|>\nAnno 1715 den 4 maj",
            id="grounded",
        ),
        pytest.param(["--task", "markdown"], "<image>\nFree OCR.", "Anno 1715 den 4 maj", id="markdown"),
    ],
)
def test_reader_set_writes_a_page_as_deepseek_ocr_2_is_fine_tuned_on_it(
    tmp_path: Path, task: list[str], prompt: str, answer: str
) -> None:
    run = _run(tmp_path / "run", LINE)
    out = tmp_path / "set" / "deepseek.jsonl"

    result = CliRunner().invoke(app, ["--run", str(run), "--model", "deepseek-ocr-2", "--out", str(out), *task])

    assert result.exit_code == 0
    assert _lines(out) == [
        {
            "messages": [
                {"role": "<|User|>", "content": prompt, "images": [str((run / IMAGE).resolve())]},
                {"role": "<|Assistant|>", "content": answer},
            ]
        }
    ]


@pytest.mark.parametrize(
    ("task", "prompt", "answer"),
    [
        pytest.param([], "<image>OCR:", "Anno 1715 den 4 maj", id="page"),
        pytest.param(
            ["--task", "spotting"],
            "<image>Spotting:",
            "Anno 1715 den 4 maj<|LOC_125|><|LOC_300|><|LOC_875|><|LOC_300|><|LOC_875|><|LOC_340|><|LOC_125|><|LOC_340|>",
            id="spotting",
        ),
    ],
)
def test_reader_set_writes_a_whole_page_as_paddleocr_vl_is_fine_tuned_on_it(
    tmp_path: Path, task: list[str], prompt: str, answer: str
) -> None:
    run = _run(tmp_path / "run", LINE)
    out = tmp_path / "set" / "paddleocr_vl.jsonl"

    result = CliRunner().invoke(app, ["--run", str(run), "--model", "paddleocr-vl", "--out", str(out), *task])

    assert result.exit_code == 0
    assert _lines(out) == [
        {
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
            "images": [str((run / IMAGE).resolve())],
        }
    ]


def test_reader_set_writes_each_element_of_a_page_as_its_crop_recognition_prompt_and_text(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", HEADING, TABLE, BODY)
    out = tmp_path / "set" / "paddleocr_vl.jsonl"

    result = CliRunner().invoke(
        app, ["--run", str(run), "--model", "paddleocr-vl", "--task", "elements", "--out", str(out)]
    )

    crops = [(tmp_path / "set" / "paddleocr_vl_crops" / f"000_page_0_{k}.png").resolve() for k in range(3)]
    texts = [
        ("OCR:", "Förteckning öfver\nJämtlands län"),
        ("Table Recognition:", "<fcel>Namn<fcel>Ort<nl><fcel>Anders<fcel>Bro<nl>"),
        ("OCR:", "Anno 1715 den 4 maj\nkom för rätten"),
    ]
    assert result.exit_code == 0
    assert _lines(out) == [
        {
            "messages": [{"role": "user", "content": f"<image>{task}"}, {"role": "assistant", "content": text}],
            "images": [str(crop)],
        }
        for (task, text), crop in zip(texts, crops, strict=True)
    ]
    page = load_rgb(run / IMAGE)
    assert [load_rgb(crop).tobytes() for crop in crops] == [
        page.crop(element.box).tobytes() for element in (HEADING, TABLE, BODY)
    ]
    assert _lines(tmp_path / "set" / "paddleocr_vl.meta.jsonl") == [
        {
            "id": f"000_page_0_{k}",
            "source": "synthetic",
            "collection": "",
            "volume": "",
            "scan": IMAGE,
            "side": None,
            "split": None,
            "kind": kind,
            "production": None,
            "element": k,
            "line": None,
        }
        for k, kind in enumerate(("heading", "table", "body"))
    ]


@pytest.mark.parametrize(
    ("elements", "image", "pages", "refusal"),
    [
        pytest.param(
            False,
            True,
            ["--run", "run"],
            "run holds no .elements.json files: draw it with diffu-generate --reader-labels",
            id="labels",
        ),
        pytest.param(
            True,
            False,
            ["--run", "run"],
            "000_page_0.elements.json labels the image '000_page_0.jpg', which the run does not hold",
            id="image",
        ),
        pytest.param(
            True,
            True,
            ["--run", "run", "--task", "grounded"],
            "paddleocr-vl is fine-tuned on page, spotting, elements",
            id="task",
        ),
        pytest.param(
            True,
            True,
            ["--run", "run", "--max-crop-pixels", "6000"],
            "page cuts no crops: give it with elements or lines",
            id="crop-pixels-of-a-page",
        ),
        pytest.param(True, True, [], "give the pages as --run or as --xml, one of them", id="no-pages"),
        pytest.param(
            True, True, ["--run", "run", "--xml", "run"], "give the pages as --run or as --xml, one of them", id="both"
        ),
        pytest.param(
            True,
            True,
            ["--run", "run", "--split-from-gt-test-lists"],
            "a run's pages are drawn, neither split by the GT's test lists, left out by a list nor of collections: "
            "give --xml",
            id="split-a-run",
        ),
        pytest.param(True, True, ["--xml", "run"], "run gives no PAGE or ALTO file", id="no-xml"),
        pytest.param(
            True,
            True,
            ["--xml", "gt", "--collection", "GT-elsewhere"],
            "gt holds no file of GT-elsewhere",
            id="unknown-collection",
        ),
    ],
)
def test_reader_set_refuses_pages_without_labels_or_images_and_a_task_its_reader_has_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    elements: bool,
    image: bool,
    pages: list[str],
    refusal: str,
) -> None:
    monkeypatch.chdir(tmp_path)  # short paths: rich wraps the error panel at 80 columns, inside a long one
    run = _run(tmp_path / "run", LINE)
    _gt_page(tmp_path / "gt", VOLUME, "a")
    if not elements:
        (run / "000_page_0.elements.json").unlink()
    if not image:
        (run / IMAGE).unlink()

    result = CliRunner().invoke(app, [*pages, "--model", "paddleocr-vl", "--out", "set.jsonl"])

    message = " ".join(result.output.replace("│", " ").split())
    assert result.exit_code == 2
    assert refusal in message
    assert not (tmp_path / "set.jsonl").exists()


def _text_line(line_id: str, box: PageBox, text: str) -> str:
    x0, y0, x1, y1 = box
    return (
        f'<TextLine id="{line_id}"><Coords points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/>'
        f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
    )


def _region(region_id: str, attributes: str, lines: list[tuple[PageBox, str]]) -> str:
    """A PAGE ``TextRegion`` of ``lines`` (each its box and text), its ``attributes`` as written (its type, if any)."""
    inner = "".join(_text_line(f"{region_id}_{k}", box, text) for k, (box, text) in enumerate(lines))
    return f'<TextRegion id="{region_id}" {attributes}>{inner}</TextRegion>'


def _page_xml(path: Path, size: tuple[int, int], regions: str) -> Path:
    """A PAGE file at ``path``, as Transkribus exports one, of a scan ``size`` px named as the file (``.jpg``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
        "<Metadata><Creator>prov=READ-COOP:name=PyLaia@TranskribusPlatform</Creator></Metadata>"
        f'<Page imageFilename="{path.stem}.jpg" imageWidth="{size[0]}" imageHeight="{size[1]}">{regions}</Page></PcGts>',
        encoding="utf-8",
    )
    return path


def _scan(path: Path, size: tuple[int, int]) -> Path:
    """A scan whose pixels tell where they are: red grows down it, green across it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    down = Image.linear_gradient("L").resize(size)
    across = Image.linear_gradient("L").transpose(Image.Transpose.ROTATE_90).resize(size)
    Image.merge("RGB", (down, across, Image.new("L", size))).save(path)
    return path


OPENING_SIZE = (2000, 1200)  # px: two pages of 1000 x 1200, the gutter at x 1000 (the only free band of the middle)
OPENING_REGIONS = (
    _region("pn", 'type="page-number"', [((880, 20, 960, 60), "12")])
    + _region("h", 'type="heading"', [((200, 80, 900, 140), "Förteckning öfver gods")])
    + _region("b", "", [((200, 200, 900, 260), "Anno 1715 den 4 maj"), ((200, 260, 900, 320), "kom för rätten")])
    + _region("m", 'type="marginalia"', [((20, 200, 180, 260), "Nota")])
    + _region("s", 'type="paragraph"', [((1040, 600, 1940, 660), "Summa 2")])
    + '<TableRegion id="t">'
    + "".join(
        f'<TableCell id="c{row}{column}" row="{row}" col="{column}">{_text_line(f"c{row}{column}", box, text)}</TableCell>'
        for row, column, box, text in (
            (0, 0, (1040, 400, 1490, 460), "Namn"),
            (0, 1, (1490, 400, 1940, 460), "Ort"),
            (1, 0, (1040, 460, 1490, 520), "Anders"),
            (1, 1, (1490, 460, 1940, 520), "Bro"),
        )
    )
    + "</TableRegion>"
)
VOLUME = Path("GT-RA_court_20260101", "Vol", "Vol")  # a collection's volume, as the GT's Transkribus export lays it out


def _opening(root: Path) -> Path:
    """An opening of the GT below ``root`` (``o.xml``), its PAGE file in its volume's ``page/`` beside its scan: on the
    left page a page number, a heading, two lines of running text and a margin note, on the right a line of running
    text and a table of two rows and two columns."""
    _scan(root / VOLUME / "o.jpg", OPENING_SIZE)
    return _page_xml(root / VOLUME / "page" / "o.xml", OPENING_SIZE, OPENING_REGIONS)


def _ids(xml: Path, root: Path) -> tuple[str, str]:
    """The ids of an opening's two pages: its PAGE file's path below ``root``, digested, and each side."""
    scan = digest(xml.relative_to(root).as_posix())
    return f"{scan}_left", f"{scan}_right"


# On each page's own grid of 0-1000: x of 1000 px as it is, y of 1200 px (20 px is 16.7, so 17)
LEFT_BLOCKS = (
    "<PAGE><BLOCK>page_number[880, 17, 960, 50]<FORK>12\n"
    "<BLOCK>title[200, 67, 900, 117]<FORK>Förteckning öfver gods\n"
    "<BLOCK>text[200, 167, 900, 267]<FORK>Anno 1715 den 4 maj\nkom för rätten\n"
    "<BLOCK>aside_text[20, 167, 180, 217]<FORK>Nota"
)
RIGHT_BLOCKS = (
    "<PAGE><BLOCK>text[40, 500, 940, 550]<FORK>Summa 2\n"
    "<BLOCK>table[40, 333, 940, 433]<FORK>"
    "<table><tr><td>Namn</td><td>Ort</td></tr><tr><td>Anders</td><td>Bro</td></tr></table>"
)


def test_reader_set_parses_each_page_of_an_opening_as_unlimited_ocr_2_on_the_page_cut_from_its_scan(
    tmp_path: Path,
) -> None:
    xml = _opening(tmp_path / "gt")
    out = tmp_path / "set" / "uocr.jsonl"

    result = CliRunner().invoke(app, ["--xml", str(tmp_path / "gt"), "--model", "unlimited-ocr-2", "--out", str(out)])

    left, right = _ids(xml, tmp_path / "gt")
    pages = [(tmp_path / "set" / "uocr_pages" / f"{page}.jpg").resolve() for page in (left, right)]
    assert result.exit_code == 0
    assert _lines(out) == [
        {
            "messages": [
                {"role": "user", "content": "<image>Multi page parsing."},
                {"role": "assistant", "content": blocks},
            ],
            "images": [str(page)],
        }
        for blocks, page in zip((LEFT_BLOCKS, RIGHT_BLOCKS), pages, strict=True)
    ]
    scan = load_rgb(tmp_path / "gt" / VOLUME / "o.jpg")
    halves = [scan.crop((0, 0, 1000, 1200)), scan.crop((1000, 0, 2000, 1200))]
    assert [load_rgb(page).size for page in pages] == [(1000, 1200), (1000, 1200)]
    assert [_mean_difference(load_rgb(page), half) < 2 for page, half in zip(pages, halves, strict=True)] == [
        True,
        True,
    ]
    meta = {"source": "real", "collection": "GT-RA_court_20260101", "volume": "Vol", "scan": "o.jpg", "split": None}
    meta |= NO_CROP
    assert _lines(tmp_path / "set" / "uocr.meta.jsonl") == [
        {"id": left, **meta, "side": "left"},
        {"id": right, **meta, "side": "right"},
    ]


def _mean_difference(a: Image.Image, b: Image.Image) -> float:
    """How far apart two images of one size are, on average, in levels of 255 (JPEG's loss is a level or so; the two
    halves of the opening's scan are some 127 apart in green)."""
    return float(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).mean())


def test_reader_set_parses_an_opening_as_unlimited_ocr_2_in_one_sample_of_its_two_pages(tmp_path: Path) -> None:
    xml = _opening(tmp_path / "gt")
    out = tmp_path / "set" / "uocr.jsonl"

    result = CliRunner().invoke(
        app, ["--xml", str(tmp_path / "gt"), "--model", "unlimited-ocr-2", "--task", "opening", "--out", str(out)]
    )

    left, right = _ids(xml, tmp_path / "gt")
    assert result.exit_code == 0
    assert _lines(out) == [
        {
            "messages": [
                {"role": "user", "content": "<image><image>Multi page parsing."},
                {"role": "assistant", "content": LEFT_BLOCKS + RIGHT_BLOCKS},
            ],
            "images": [str((tmp_path / "set" / "uocr_pages" / f"{page}.jpg").resolve()) for page in (left, right)],
        }
    ]
    assert [(row["id"], row["side"]) for row in _lines(tmp_path / "set" / "uocr.meta.jsonl")] == [
        (digest(xml.relative_to(tmp_path / "gt").as_posix()), None)
    ]


def test_reader_set_reads_each_page_of_an_opening_whole_as_minicpm_ocr_reads_it(tmp_path: Path) -> None:
    _opening(tmp_path / "gt")
    out = tmp_path / "set" / "minicpm.jsonl"

    result = CliRunner().invoke(
        app, ["--xml", str(tmp_path / "gt"), "--model", "minicpm-ocr", "--task", "page", "--out", str(out)]
    )

    assert result.exit_code == 0
    assert [(row["messages"], len(row["images"])) for row in _lines(out)] == [
        (
            [
                {"role": "user", "content": "<image>Text Recognition:"},
                {"role": "assistant", "content": text},
            ],
            1,
        )
        for text in (
            "12\n\nFörteckning öfver gods\n\nAnno 1715 den 4 maj\nkom för rätten\n\nNota",
            "Summa 2\n\nNamn\nOrt\nAnders\nBro",
        )
    ]


def test_reader_set_writes_each_element_of_an_opening_as_minicpm_ocr_reads_its_crop(tmp_path: Path) -> None:
    xml = _opening(tmp_path / "gt")
    out = tmp_path / "set" / "minicpm.jsonl"

    result = CliRunner().invoke(app, ["--xml", str(tmp_path / "gt"), "--model", "minicpm-ocr", "--out", str(out)])

    left, right = _ids(xml, tmp_path / "gt")
    elements = [  # each element's page, place in its reading order, box on the scan, prompt and answer
        (left, 0, (880, 20, 960, 60), "Text Recognition:", "12"),
        (left, 1, (200, 80, 900, 140), "Text Recognition:", "Förteckning öfver gods"),
        (left, 2, (200, 200, 900, 320), "Text Recognition:", "Anno 1715 den 4 maj\nkom för rätten"),
        (left, 3, (20, 200, 180, 260), "Text Recognition:", "Nota"),
        (right, 0, (1040, 600, 1940, 660), "Text Recognition:", "Summa 2"),
        (
            right,
            1,
            (1040, 400, 1940, 520),
            "Table Recognition:",
            "<otsl><fcel>Namn<fcel>Ort<nl><fcel>Anders<fcel>Bro<nl></otsl>",
        ),
    ]
    crops = [(tmp_path / "set" / "minicpm_crops" / f"{page}_{k}.png").resolve() for page, k, *_ in elements]
    assert result.exit_code == 0
    assert _lines(out) == [
        {
            "messages": [{"role": "user", "content": f"<image>{prompt}"}, {"role": "assistant", "content": answer}],
            "images": [str(crop)],
        }
        for (*_, prompt, answer), crop in zip(elements, crops, strict=True)
    ]
    scan = load_rgb(tmp_path / "gt" / VOLUME / "o.jpg")
    assert [load_rgb(crop).tobytes() for crop in crops] == [scan.crop(box).tobytes() for _, _, box, *_ in elements]
    assert [row["id"] for row in _lines(tmp_path / "set" / "minicpm.meta.jsonl")] == [
        f"{page}_{k}" for page, k, *_ in elements
    ]


def test_reader_set_writes_each_text_line_of_an_opening_as_minicpm_ocr_reads_a_line_crop(tmp_path: Path) -> None:
    xml = _opening(tmp_path / "gt")
    out = tmp_path / "set" / "minicpm_lines.jsonl"

    result = CliRunner().invoke(
        app, ["--xml", str(tmp_path / "gt"), "--model", "minicpm-ocr", "--task", "lines", "--out", str(out)]
    )

    left, right = _ids(xml, tmp_path / "gt")
    lines = [  # each line's page, element, place in it, kind, box on the scan and text
        (left, 0, 0, "page-number", (880, 20, 960, 60), "12"),
        (left, 1, 0, "heading", (200, 80, 900, 140), "Förteckning öfver gods"),
        (left, 2, 0, "body", (200, 200, 900, 260), "Anno 1715 den 4 maj"),
        (left, 2, 1, "body", (200, 260, 900, 320), "kom för rätten"),
        (left, 3, 0, "marginalia", (20, 200, 180, 260), "Nota"),
        (right, 0, 0, "body", (1040, 600, 1940, 660), "Summa 2"),
        (right, 1, 0, "table", (1040, 400, 1490, 460), "Namn"),
        (right, 1, 1, "table", (1490, 400, 1940, 460), "Ort"),
        (right, 1, 2, "table", (1040, 460, 1490, 520), "Anders"),
        (right, 1, 3, "table", (1490, 460, 1940, 520), "Bro"),
    ]
    crops = [(tmp_path / "set" / "minicpm_lines_lines" / f"{page}_{k}_{j}.jpg").resolve() for page, k, j, *_ in lines]
    assert result.exit_code == 0
    assert _lines(out) == [
        {
            "messages": [
                {"role": "user", "content": "<image>Text Recognition:"},
                {"role": "assistant", "content": text},
            ],
            "images": [str(crop)],
        }
        for (*_, text), crop in zip(lines, crops, strict=True)
    ]
    scan = load_rgb(tmp_path / "gt" / VOLUME / "o.jpg")
    assert [load_rgb(crop).size for crop in crops] == [(x1 - x0, y1 - y0) for *_, (x0, y0, x1, y1), _ in lines]
    assert all(
        _mean_difference(load_rgb(crop), scan.crop(box)) < 2 for crop, (*_, box, _) in zip(crops, lines, strict=True)
    )
    common = {"source": "real", "collection": "GT-RA_court_20260101", "volume": "Vol", "scan": "o.jpg", "split": None}
    assert _lines(tmp_path / "set" / "minicpm_lines.meta.jsonl") == [
        {
            "id": f"{page}_{k}_{j}",
            **common,
            "side": "left" if page == left else "right",
            "kind": kind,
            "production": "handwritten",
            "element": k,
            "line": j,
        }
        for page, k, j, kind, *_ in lines
    ]


def test_reader_set_scales_line_crops_to_at_most_max_crop_pixels_and_records_the_lines_it_leaves_out(
    tmp_path: Path,
) -> None:
    body = PageElement(
        kind=LineKind.BODY,
        box=(100, 300, 700, 420),
        lines=[
            ElementLine(text="Anno 1715 den 4 maj", box=(100, 300, 700, 340)),  # 600 x 40 = 24,000 px
            ElementLine(text=" ", box=(100, 340, 700, 380)),
            ElementLine(text="kom för rätten", box=(100, 380, 700, 420), tag="unclear"),
            ElementLine(text="a stray box", box=(100, 420, 700, 427)),  # 7 px high
            ElementLine(text="a box with no width", box=(100, 430, 100, 470)),
        ],
    )
    run = _run(tmp_path / "run", body)
    out = tmp_path / "set" / "lines.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "--run",
            str(run),
            "--model",
            "minicpm-ocr",
            "--task",
            "lines",
            "--max-crop-pixels",
            "6000",
            "--out",
            str(out),
        ],
    )

    crop = (tmp_path / "set" / "lines_lines" / "000_page_0_0_0.jpg").resolve()
    assert result.exit_code == 0
    assert [row["images"] for row in _lines(out)] == [[str(crop)]]
    assert load_rgb(crop).size == (300, 20)  # a quarter of the pixels: half each side
    assert _lines(tmp_path / "set" / "lines.skipped.jsonl") == [
        {
            "labels": str(run / "000_page_0.elements.json"),
            "side": None,
            "reason": reason,
            "error": "",
            "id": f"000_page_0_0_{j}",
        }
        for j, reason in ((1, "empty-line"), (2, "unclear-line"), (3, "small-line"), (4, "small-line"))
    ]


def test_reader_set_masks_a_line_to_its_outline_and_leaves_out_doubts_and_small_lines_as_the_trocr_baseline_does(
    tmp_path: Path,
) -> None:
    outline = [(100, 300), (700, 300), (100, 340)]  # the box's upper left half
    body = PageElement(
        kind=LineKind.BODY,
        box=(100, 300, 700, 460),
        lines=[
            ElementLine(text="Anno 1715 den 4 maj", box=(100, 300, 700, 340), polygon=outline),
            ElementLine(text="hwilket (?) bewistes", box=(100, 340, 700, 380)),
            ElementLine(text="Nilsson ibidem. Desse bekänn[e]r", box=(100, 380, 700, 420)),
            ElementLine(text="nine px high", box=(100, 420, 700, 429)),
        ],
    )
    run = _run(tmp_path / "run", body)
    out = tmp_path / "set" / "lines.jsonl"

    result = CliRunner().invoke(
        app,
        [
            *("--run", str(run), "--model", "minicpm-ocr", "--task", "lines", "--out", str(out)),
            *("--mask-lines", "--reject-uncertain", "--min-line-side", "10"),
        ],
    )

    crop = load_rgb(tmp_path / "set" / "lines_lines" / "000_page_0_0_0.jpg")
    page = load_rgb(run / IMAGE).crop((100, 300, 700, 340))
    assert result.exit_code == 0
    assert crop.size == (600, 40)
    assert all(abs(a - b) < 40 for a, b in zip(crop.getpixel((20, 5)), page.getpixel((20, 5)), strict=True))
    assert crop.getpixel((580, 35)) == pytest.approx((255, 255, 255), abs=8)  # below the outline: white
    assert [row["reason"] for row in _lines(tmp_path / "set" / "lines.skipped.jsonl")] == [
        "uncertain-line",
        "uncertain-line",
        "small-line",
    ]


def test_reader_set_scales_an_element_crop_past_max_crop_pixels_and_keeps_one_within_it_as_cut(tmp_path: Path) -> None:
    note = PageElement(  # 100 x 40 = 4,000 px
        kind=LineKind.BODY, box=(100, 500, 200, 540), lines=[ElementLine(text="Nota", box=(100, 500, 200, 540))]
    )
    run = _run(tmp_path / "run", BODY, note)  # BODY: 600 x 80 = 48,000 px
    out = tmp_path / "set" / "minicpm.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "--run",
            str(run),
            "--model",
            "minicpm-ocr",
            "--task",
            "elements",
            "--max-crop-pixels",
            "12000",
            "--out",
            str(out),
        ],
    )

    crops = [tmp_path / "set" / "minicpm_crops" / f"000_page_0_{k}.png" for k in range(2)]
    assert result.exit_code == 0
    assert [load_rgb(crop).size for crop in crops] == [(300, 40), (100, 40)]  # a quarter of BODY's pixels: half a side
    assert load_rgb(crops[1]).tobytes() == load_rgb(run / IMAGE).crop(note.box).tobytes()


def _gt_page(root: Path, volume: Path, name: str) -> Path:
    """A single page of the GT below ``root``: its PAGE file in ``volume``'s ``page/`` and its 100 x 200 px scan."""
    _scan(root / volume / f"{name}.jpg", (100, 200))
    return _page_xml(root / volume / "page" / f"{name}.xml", (100, 200), _region("r", "", [((10, 10, 90, 40), name)]))


KEPT = Path("GT-RA_landskapshandlingar_20260220", "L", "L")  # the GT holds this collection twice: the export kept
TWIN = Path("GT-RA_landskapshandlingar_20260227", "L", "L")  # and its second export, the same scans


def test_reader_set_splits_the_gt_by_its_test_lists_its_second_export_counted_and_left_out(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    pages = {name: _gt_page(gt, VOLUME, name) for name in ("a", "b", "e")} | {
        name: _gt_page(gt, KEPT, name) for name in ("c", "d")
    }
    twins = [_gt_page(gt, TWIN, name) for name in ("c", "d")]
    (gt / VOLUME / "page" / "test_0025").write_text("b.xml\n", encoding="utf-8")  # nested lists: 0025 in 0100
    (gt / VOLUME / "page" / "test_0100").write_text("b.xml\ne.xml", encoding="utf-8")
    (gt / TWIN / "page" / "test_0100").write_text("d.xml\n", encoding="utf-8")  # a test page of the second export only
    out = tmp_path / "set" / "gt.jsonl"

    result = CliRunner().invoke(
        app,
        ["--xml", str(gt), "--model", "minicpm-ocr", "--task", "page", "--split-from-gt-test-lists", "--out", str(out)],
    )

    def meta(name: str, collection: str, volume: str, split: str) -> dict[str, object]:
        identity = digest(pages[name].relative_to(gt).as_posix())
        return {
            "id": identity,
            "source": "real",
            "collection": collection,
            "volume": volume,
            "scan": f"{name}.jpg",
            "side": None,
            "split": split,
            **NO_CROP,
        }

    court, kept = "GT-RA_court_20260101", "GT-RA_landskapshandlingar_20260220"
    assert result.exit_code == 0
    assert _lines(tmp_path / "set" / "gt.meta.jsonl") == [
        meta("a", court, "Vol", "train"),
        meta("c", kept, "L", "train"),
    ]
    assert _lines(tmp_path / "set" / "gt.test.meta.jsonl") == [
        meta("b", court, "Vol", "test"),
        meta("e", court, "Vol", "test"),
        meta("d", kept, "L", "test"),
    ]
    assert [row["images"] for row in _lines(tmp_path / "set" / "gt.test.jsonl")] == [
        [str(gt / volume / f"{name}.jpg")] for name, volume in (("b", VOLUME), ("e", VOLUME), ("d", KEPT))
    ]
    assert _lines(tmp_path / "set" / "gt.skipped.jsonl") == [
        {"labels": str(twin), "side": None, "reason": "duplicate", "error": "", "id": ""} for twin in twins
    ]


def test_reader_set_masks_a_gt_line_to_the_outline_its_page_file_gives(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    _scan(gt / VOLUME / "a.jpg", (100, 200))
    triangle = (  # the box (10, 10, 90, 40)'s upper left half: its lower right corner lies outside
        '<TextRegion id="r"><TextLine id="r_0"><Coords points="10,10 90,10 10,40"/>'
        "<TextEquiv><Unicode>Anno 1715</Unicode></TextEquiv></TextLine></TextRegion>"
    )
    _page_xml(gt / VOLUME / "page" / "a.xml", (100, 200), triangle)
    out = tmp_path / "set" / "lines.jsonl"

    result = CliRunner().invoke(
        app, ["--xml", str(gt), "--model", "minicpm-ocr", "--task", "lines", "--mask-lines", "--out", str(out)]
    )

    crop = load_rgb(Path(_lines(out)[0]["images"][0]))
    assert result.exit_code == 0
    assert crop.size == (80, 30)
    assert crop.getpixel((77, 27)) == pytest.approx((255, 255, 255), abs=8)
    assert crop.getpixel((3, 3)) != pytest.approx((255, 255, 255), abs=8)


def test_reader_set_splits_the_gt_by_the_test_list_it_is_given_alone(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    for name in ("a", "b", "e"):
        _gt_page(gt, VOLUME, name)
    (gt / VOLUME / "page" / "test_0050").write_text("b.xml\n", encoding="utf-8")  # nested: 0050 in 0100
    (gt / VOLUME / "page" / "test_0100").write_text("b.xml\ne.xml", encoding="utf-8")
    out = tmp_path / "set" / "gt.jsonl"

    result = CliRunner().invoke(
        app,
        [
            *("--xml", str(gt), "--model", "minicpm-ocr", "--task", "page", "--out", str(out)),
            *("--split-from-gt-test-lists", "--gt-test-list", "test_0050"),
        ],
    )

    assert result.exit_code == 0
    assert [row["scan"] for row in _lines(tmp_path / "set" / "gt.meta.jsonl")] == ["a.jpg", "e.jpg"]
    assert [row["scan"] for row in _lines(tmp_path / "set" / "gt.test.meta.jsonl")] == ["b.jpg"]
    assert (
        json.loads((tmp_path / "set" / "gt.options.json").read_text(encoding="utf-8"))["gt_test_lists"] == "test_0050"
    )


@pytest.mark.parametrize(
    ("choice", "selection"),
    [
        pytest.param(
            ["--exclude-collection", KEPT.parts[0]],
            {"include": [], "exclude": [KEPT.parts[0]], "seed": None},
            id="the-other-left-out",
        ),
        pytest.param(
            ["--collection", VOLUME.parts[0]],
            {"include": [VOLUME.parts[0]], "exclude": [], "seed": None},
            id="only-this-one",
        ),
    ],
)
def test_reader_set_takes_the_collections_it_is_given_caps_their_training_side_and_records_its_options(
    tmp_path: Path, choice: list[str], selection: dict[str, object]
) -> None:
    gt = tmp_path / "gt"
    court = {name: _gt_page(gt, VOLUME, name) for name in ("a", "b", "c", "x", "y", "z")}
    (gt / VOLUME / "page" / "test_0100").write_text("x.xml\ny.xml\nz.xml\n", encoding="utf-8")
    _gt_page(gt, KEPT, "d")
    out = tmp_path / "set" / "court.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "--xml",
            str(gt),
            "--model",
            "minicpm-ocr",
            "--task",
            "page",
            "--split-from-gt-test-lists",
            *choice,
            "--max-per-collection",
            "2",
            "--out",
            str(out),
        ],
    )

    def images(path: Path) -> list[list[str]]:
        return [row["images"] for row in _lines(path)]

    assert result.exit_code == 0
    assert images(out) == [[str(gt / VOLUME / f"{name}.jpg")] for name in ("a", "b")]
    assert images(out.with_name("court.test.jsonl")) == [[str(gt / VOLUME / f"{name}.jpg")] for name in "xyz"]
    assert _lines(tmp_path / "set" / "court.skipped.jsonl") == [
        {"labels": str(court["c"]), "side": None, "reason": "capped", "error": "", "id": ""}
    ]
    assert json.loads((tmp_path / "set" / "court.options.json").read_text(encoding="utf-8")) == {
        "model": "minicpm-ocr",
        "task": "page",
        "run": None,
        "xml": str(gt.resolve()),
        "split_from_gt_test_lists": True,
        "gt_test_lists": "test_*",
        "exclude": None,
        "selection": selection,
        "limits": {
            "max_crop_pixels": None,
            "max_per_collection": 2,
            "line_rules": {"mask": False, "min_side": 8, "reject_uncertain": False},
        },
        "samples": {"train": 2, "test": 3},
        "pages": 5,
        "scans": 5,
        "skipped": {"capped": 1},
    }


def test_reader_set_reads_the_scans_in_the_order_its_seed_shuffles_them(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    names = ["a", "b", "c", "d", "e", "f"]
    for name in names:
        _gt_page(gt, VOLUME, name)
    out = tmp_path / "set" / "seeded.jsonl"

    result = CliRunner().invoke(
        app, ["--xml", str(gt), "--model", "minicpm-ocr", "--task", "page", "--seed", "3", "--out", str(out)]
    )

    shuffled = list(names)  # the files in path order, shuffled by the seed
    random.Random(3).shuffle(shuffled)
    assert result.exit_code == 0
    assert shuffled != names
    scans = [row["scan"] for row in _lines(tmp_path / "set" / "seeded.meta.jsonl")]
    assert scans == [f"{name}.jpg" for name in shuffled]


def test_reader_set_leaves_out_the_pages_a_list_names_by_path_or_by_page_pair_id(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    single, opening = _gt_page(gt, VOLUME, "a"), _opening(gt)
    left, right = _ids(opening, gt)
    exclude = tmp_path / "trained_on.txt"
    exclude.write_text(f"# the generator's pages\n{single.relative_to(gt).as_posix()}\n\n{right}\n", encoding="utf-8")
    out = tmp_path / "set" / "uocr.jsonl"

    result = CliRunner().invoke(
        app, ["--xml", str(gt), "--model", "unlimited-ocr-2", "--exclude", str(exclude), "--out", str(out)]
    )

    assert result.exit_code == 0
    assert [row["id"] for row in _lines(tmp_path / "set" / "uocr.meta.jsonl")] == [left]
    assert _lines(tmp_path / "set" / "uocr.skipped.jsonl") == [
        {"labels": str(single), "side": None, "reason": "excluded", "error": "", "id": ""},
        {"labels": str(opening), "side": "right", "reason": "excluded", "error": "", "id": ""},
    ]


def _drawn(lines: list[tuple[str, LineKind, Rect]], image: str) -> PageLabels:
    """A drawn 1000 x 1000 px page's labels: each line its text, kind and box, in reading order, one block a kind."""
    return PageLabels(
        width=1000,
        height=1000,
        image=image,
        lines=[
            LineLabel(text=text, polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], region=kind, block=kind, order=k)
            for k, (text, kind, (x0, y0, x1, y1)) in enumerate(lines)
        ],
    )


@pytest.mark.parametrize(
    "listed",
    [
        pytest.param(None, id="folder"),
        pytest.param("# the run's files\np2.alto.xml\n\np1.alto.xml\np1.page.xml\n", id="list"),
    ],
)
def test_reader_set_reads_a_drawn_runs_pages_from_their_page_files_and_from_alto_where_a_page_has_no_other(
    tmp_path: Path, listed: str | None
) -> None:
    files = tmp_path / "files"  # as diffu-run-export writes a run: <id>.jpg, <id>.alto.xml and <id>.page.xml
    lines = [
        ("Anno 1650", LineKind.HEADING, (100, 100, 900, 150)),
        ("Tå laga tingh", LineKind.BODY, (100, 200, 900, 250)),
    ]
    for page in ("p1", "p2"):
        _scan(files / f"{page}.jpg", (1000, 1000))
        (files / f"{page}.alto.xml").write_text(to_alto(_drawn(lines, f"{page}.jpg")), encoding="utf-8")
    (files / "p1.page.xml").write_text(to_page_xml(_drawn(lines, "p1.jpg")), encoding="utf-8")
    if listed is not None:  # a list of the files, read from its folder
        (files / "pages.txt").write_text(listed, encoding="utf-8")
    out = tmp_path / "set" / "uocr.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "--xml",
            str(files if listed is None else files / "pages.txt"),
            "--model",
            "unlimited-ocr-2",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert [row["messages"][1]["content"] for row in _lines(out)] == [
        "<PAGE><BLOCK>title[100, 100, 900, 150]<FORK>Anno 1650\n<BLOCK>text[100, 200, 900, 250]<FORK>Tå laga tingh",
        "<PAGE><BLOCK>text[100, 100, 900, 150]<FORK>Anno 1650\n<BLOCK>text[100, 200, 900, 250]<FORK>Tå laga tingh",
    ]
    assert _lines(tmp_path / "set" / "uocr.meta.jsonl") == [
        {
            "id": digest(f"{page}.{kind}.xml"),
            "source": "synthetic",
            "collection": "",
            "volume": "",
            "scan": f"{page}.jpg",
            "side": None,
            "split": None,
            **NO_CROP,
        }
        for page, kind in (("p1", "page"), ("p2", "alto"))
    ]
    assert _lines(tmp_path / "set" / "uocr.skipped.jsonl") == []


def test_reader_set_records_each_file_scan_or_page_it_leaves_out_and_why(
    tmp_path: Path, unsplit_spread_page: Path
) -> None:
    folder = tmp_path / "pages"
    _gt_page(folder, Path("."), "good")  # its PAGE file in page/, beside its scan
    blank = _page_xml(folder / "blank.xml", (100, 200), "")
    _scan(folder / "blank.jpg", (100, 200))
    off = _page_xml(folder / "off.xml", (100, 200), _region("r", "", [((150, 10, 190, 40), "off the scan")]))
    _scan(folder / "off.jpg", (100, 200))
    small = _page_xml(folder / "small.xml", (100, 200), _region("r", "", [((10, 10, 90, 40), "small")]))
    _scan(folder / "small.jpg", (50, 100))
    spread = folder / "spread.xml"
    spread.write_bytes(unsplit_spread_page.read_bytes())
    _scan(folder / "spread.jpg", (2500, 1200))
    lonely = _page_xml(folder / "lonely.xml", (100, 200), _region("r", "", [((10, 10, 90, 40), "no scan")]))
    broken = folder / "broken.xml"
    broken.write_text("<PcGts", encoding="utf-8")
    sizeless = _page_xml(folder / "sizeless.xml", (100, 200), _region("r", "", [((10, 10, 90, 40), "sizeless")]))
    sizeless.write_text(sizeless.read_text(encoding="utf-8").replace(' imageWidth="100"', ""), encoding="utf-8")
    _scan(folder / "sizeless.jpg", (100, 200))
    torn = _page_xml(folder / "torn.xml", OPENING_SIZE, OPENING_REGIONS)  # an opening, its pages cut from a torn scan
    whole = _scan(folder / "torn.jpg", OPENING_SIZE).read_bytes()
    (folder / "torn.jpg").write_bytes(whole[: len(whole) // 2])  # its size still reads, its pixels do not
    with pytest.raises(OSError) as tear:
        load_rgb(folder / "torn.jpg")
    (folder / "mets.xml").write_text('<mets xmlns="http://www.loc.gov/METS/"/>', encoding="utf-8")  # passed over
    out = tmp_path / "set" / "minicpm.jsonl"

    result = CliRunner().invoke(
        app, ["--xml", str(folder), "--model", "minicpm-ocr", "--task", "page", "--out", str(out)]
    )

    assert result.exit_code == 0
    assert [row["images"] for row in _lines(out)] == [[str(folder / "good.jpg")]]
    assert _lines(tmp_path / "set" / "minicpm.skipped.jsonl") == [
        {"labels": str(labels), "side": None, "reason": reason, "error": error, "id": ""}
        for labels, reason, error in (
            (broken, "unreadable", "ParseError: unclosed token: line 1, column 0"),
            (lonely, "no-scan", ""),
            (blank, "no-text", ""),
            (off, "no-text", ""),
            (sizeless, "unreadable", f"ValueError: {sizeless}: the Page element has no width and height"),
            (small, "size", "small.jpg is (50, 100) px, its file gives (100, 200)"),
            (spread, "spread", ""),
            (torn, "unreadable", f"OSError: {tear.value}"),
        )
    ]
