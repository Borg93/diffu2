"""A page reader's fine-tuning set from a diffu-generate run drawn with ``--reader-labels``: each image of the run and its
``<name>.elements.json`` (:class:`diffu_page.labels.PageElements`) written as JSON lines in the reader's own
conversation format, image paths absolute. By default each reader is taught a page whole, as pages will be sent to it.

- ``deepseek-ocr-2``: a sample per image, as DeepSeek-OCR 2 is fine-tuned (Unsloth's ``<|User|>`` and ``<|Assistant|>``
  turns, the user's naming the image). ``--task grounded`` (the default) asks with its prompt for a document and
  answers with the grounded Markdown (:func:`diffu_page.labels.to_grounded_markdown`); ``--task markdown`` asks with
  its prompt for a document without layouts (Free OCR) and answers with the Markdown.
- ``paddleocr-vl``: as PaddleFormers fine-tunes PaddleOCR-VL (``messages`` and ``images``). ``--task page`` (the
  default) a sample per image, asked ``OCR:`` and answered with the Markdown, as DeepSeek-OCR 2's Free OCR is;
  ``--task spotting`` a sample per image, asked ``Spotting:`` and answered with each line's text and box
  (:func:`diffu_page.labels.to_spotting`); ``--task elements`` a sample per layout element, for its recogniser: the
  element's crop, cut from its image into ``<out stem>_crops/``, its recognition prompt (``OCR:`` or
  ``Table Recognition:``) and the text it reads there (a table as OTSL).

    diffu-reader-set --run runs/pages --model deepseek-ocr-2 --out data/readers/deepseek_ocr_2.jsonl
    diffu-reader-set --run runs/pages --model paddleocr-vl --out data/readers/paddleocr_vl.jsonl
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from pydantic import BaseModel

from diffu_page.labels import (
    ELEMENTS_FILE,
    OCR_PROMPT,
    SPOTTING_PROMPT,
    PageElements,
    to_grounded_markdown,
    to_markdown,
    to_spotting,
)
from diffu_page.output import console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class Reader(StrEnum):
    """The page readers a set is written for."""

    DEEPSEEK_OCR_2 = "deepseek-ocr-2"
    PADDLEOCR_VL = "paddleocr-vl"


class Task(StrEnum):
    """What a set teaches its reader: a whole page grounded (DeepSeek-OCR 2), as Markdown alone (either reader) or its
    lines spotted (PaddleOCR-VL), or each element's crop (PaddleOCR-VL's recogniser)."""

    GROUNDED = "grounded"
    MARKDOWN = "markdown"
    PAGE = "page"
    SPOTTING = "spotting"
    ELEMENTS = "elements"


TASKS: dict[Reader, tuple[Task, ...]] = {  # the tasks each reader is fine-tuned on, its default first
    Reader.DEEPSEEK_OCR_2: (Task.GROUNDED, Task.MARKDOWN),
    Reader.PADDLEOCR_VL: (Task.PAGE, Task.SPOTTING, Task.ELEMENTS),
}

# Each page task's prompt: DeepSeek-OCR 2's for a document and for one without layouts, without the trailing space its
# model card shows (its own inference strips a turn's content, so the model never reads one there); PaddleOCR-VL's after
# the image placeholder, as PaddleFormers' samples write it.
PAGE_PROMPTS: dict[Task, str] = {
    Task.GROUNDED: "<image>\n<|grounding|>Convert the document to markdown.",
    Task.MARKDOWN: "<image>\nFree OCR.",
    Task.PAGE: f"<image>{OCR_PROMPT}",
    Task.SPOTTING: f"<image>{SPOTTING_PROMPT}",
}
PAGE_ANSWERS: dict[Task, Callable[[PageElements], str]] = {
    Task.GROUNDED: to_grounded_markdown,
    Task.MARKDOWN: to_markdown,
    Task.PAGE: to_markdown,
    Task.SPOTTING: to_spotting,
}


class Turn(BaseModel):
    """One turn of a training conversation: who speaks, what, and (DeepSeek-OCR 2's user turn) the images it shows."""

    role: str
    content: str
    images: list[str] | None = None


class Sample(BaseModel):
    """One line of a set: its conversation, and (PaddleOCR-VL's) the images it shows."""

    messages: list[Turn]
    images: list[str] | None = None


def run_pages(run: Path) -> list[tuple[Path, PageElements]]:
    """Each image of the run, as an absolute path, with its elements, in the order of their files' names.

    Raises:
        typer.BadParameter: If the run holds no elements files, or one names an image the run does not hold.
    """
    pages: list[tuple[Path, PageElements]] = []
    for path in sorted(run.glob(f"*{ELEMENTS_FILE}")):
        page = PageElements.model_validate_json(path.read_text(encoding="utf-8"))
        image = (run / page.image).resolve()
        if not page.image or not image.is_file():
            raise typer.BadParameter(
                f"{path.name} labels the image {page.image!r}, which the run does not hold", param_hint="--run"
            )
        pages.append((image, page))
    if not pages:
        raise typer.BadParameter(
            f"{run} holds no {ELEMENTS_FILE} files: draw it with diffu-generate --reader-labels", param_hint="--run"
        )
    return pages


def deepseek_sample(page: PageElements, image: Path, task: Task) -> Sample:
    """The page drawn as ``image`` as DeepSeek-OCR 2 is fine-tuned on it: asked with ``task``'s prompt in a turn that
    shows the image, answered with the page as ``task`` writes it."""
    return Sample(
        messages=[
            Turn(role="<|User|>", content=PAGE_PROMPTS[task], images=[str(image)]),
            Turn(role="<|Assistant|>", content=PAGE_ANSWERS[task](page)),
        ]
    )


def paddleocr_vl_sample(page: PageElements, image: Path, task: Task) -> Sample:
    """The page drawn as ``image`` as PaddleOCR-VL is fine-tuned on it: shown the image, asked with ``task``'s prompt,
    answered with the page as ``task`` writes it."""
    return Sample(
        messages=[
            Turn(role="user", content=PAGE_PROMPTS[task]),
            Turn(role="assistant", content=PAGE_ANSWERS[task](page)),
        ],
        images=[str(image)],
    )


PAGE_SAMPLES: dict[Reader, Callable[[PageElements, Path, Task], Sample]] = {  # a whole page, as each reader is taught
    Reader.DEEPSEEK_OCR_2: deepseek_sample,
    Reader.PADDLEOCR_VL: paddleocr_vl_sample,
}


def cut_elements(page: PageElements, image: Path, crops: Path) -> list[Path]:
    """Each element of ``page`` cut from ``image`` at its box and saved in ``crops`` as ``<image stem>_<k>.png`` (``k``
    its place in reading order): their paths."""
    paths: list[Path] = []
    with Image.open(image) as scan:
        for k, element in enumerate(page.elements):
            path = crops / f"{image.stem}_{k}.png"
            scan.crop(element.box).save(path)
            paths.append(path)
    return paths


def element_samples(page: PageElements, crops: list[Path]) -> list[Sample]:
    """Each element of ``page`` as PaddleOCR-VL's recogniser is fine-tuned on it: shown its crop (``crops``, in the
    elements' order), asked with its recognition prompt, answered with the text it reads there."""
    return [
        Sample(
            messages=[
                Turn(role="user", content=f"<image>{element.task}"),
                Turn(role="assistant", content=element.text),
            ],
            images=[str(crop)],
        )
        for element, crop in zip(page.elements, crops, strict=True)
    ]


@app.command()
def reader_set(
    run: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="a diffu-generate run drawn with --reader-labels")
    ],
    model: Annotated[Reader, typer.Option(help="the page reader to fine-tune")],
    out: Annotated[
        Path,
        typer.Option(
            help="the JSON-lines file to write; with --task elements the crops go to <out stem>_crops beside it"
        ),
    ],
    task: Annotated[
        Task | None,
        typer.Option(
            help="deepseek-ocr-2: grounded (the grounded Markdown; default) or markdown (Free OCR: the Markdown alone); "
            "paddleocr-vl: page (OCR: the Markdown; default), spotting (each line's text and box) or elements (each "
            "element's crop, for its recogniser)"
        ),
    ] = None,
) -> None:
    """Write the pages of a run as a page reader's fine-tuning set."""
    task = task or TASKS[model][0]
    if task not in TASKS[model]:
        raise typer.BadParameter(f"{model} is fine-tuned on {', '.join(TASKS[model])}", param_hint="--task")
    pages = run_pages(run)
    if task is Task.ELEMENTS:
        crops = (out.parent / f"{out.stem}_crops").resolve()
        crops.mkdir(parents=True, exist_ok=True)
        samples = [
            sample for image, page in pages for sample in element_samples(page, cut_elements(page, image, crops))
        ]
    else:
        samples = [PAGE_SAMPLES[model](page, image, task) for image, page in pages]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{sample.model_dump_json(exclude_none=True)}\n" for sample in samples), encoding="utf-8")
    console.print(f"{len(samples)} samples from {len(pages)} images -> {out}")


if __name__ == "__main__":
    app()
