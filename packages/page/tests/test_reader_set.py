"""A page reader's fine-tuning set from a run's images and their elements files: a sample per page in DeepSeek-OCR 2's
conversation and in PaddleOCR-VL's (the page as Markdown, or its lines spotted), or a sample per element (its crop,
recognition prompt and text) for PaddleOCR-VL's recogniser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from diffu_page.cli.reader_set import app
from diffu_page.labels import ElementLine, PageElement, PageElements, to_elements_json
from diffu_page.layout import CellRef, LineKind, Production
from diffu_page.output import load_rgb

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


def _lines(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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


@pytest.mark.parametrize(
    ("elements", "image", "options", "refusal"),
    [
        pytest.param(
            False,
            True,
            [],
            "run holds no .elements.json files: draw it with diffu-generate --reader-labels",
            id="labels",
        ),
        pytest.param(
            True,
            False,
            [],
            "000_page_0.elements.json labels the image '000_page_0.jpg', which the run does not hold",
            id="image",
        ),
        pytest.param(
            True, True, ["--task", "grounded"], "paddleocr-vl is fine-tuned on page, spotting, elements", id="task"
        ),
    ],
)
def test_reader_set_refuses_a_run_without_reader_labels_or_their_images_and_a_task_its_reader_has_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    elements: bool,
    image: bool,
    options: list[str],
    refusal: str,
) -> None:
    monkeypatch.chdir(tmp_path)  # short paths: rich wraps the error panel at 80 columns, inside a long one
    run = _run(tmp_path / "run", LINE)
    if not elements:
        (run / "000_page_0.elements.json").unlink()
    if not image:
        (run / IMAGE).unlink()

    result = CliRunner().invoke(app, ["--run", "run", "--model", "paddleocr-vl", "--out", "set.jsonl", *options])

    message = " ".join(result.output.replace("│", " ").split())
    assert result.exit_code == 2
    assert refusal in message
    assert not (tmp_path / "set.jsonl").exists()
