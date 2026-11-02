"""A page reader's fine-tuning set: pages written as JSON lines in the reader's own conversation format, image paths
absolute, from a diffu-generate run drawn with ``--reader-labels`` (``--run``: each image and its
``<name>.elements.json``, :class:`diffu_page.page.labels.PageElements`) or from PAGE or ALTO files with their scans
(``--xml``: a folder of them, the GT's or a ``diffu-run-export``'s, or a list of them; an opening's two pages each
cut from it, :mod:`diffu_page.data.reader_pages`). Beside the set, ``<out stem>.meta.jsonl`` gives each sample's line
of meta (:class:`diffu_page.data.reader_pages.SampleMeta`: its id, real or synthetic, collection, volume, scan, side of
an opening and split) and ``<out stem>.skipped.jsonl`` each file, scan or page left out and why. By default each reader
is taught a page whole, as pages will be sent to it.

- ``deepseek-ocr-2``: a sample per page, as DeepSeek-OCR 2 is fine-tuned (Unsloth's ``<|User|>`` and
  ``<|Assistant|>`` turns, the user's naming the image). ``--task grounded`` (the default) asks with its prompt for a
  document and answers with the grounded Markdown (:func:`diffu_page.page.labels.to_grounded_markdown`); ``--task
  markdown`` asks with its prompt for a document without layouts (Free OCR) and answers with the Markdown.
- ``paddleocr-vl``: as PaddleFormers fine-tunes PaddleOCR-VL (``messages`` and ``images``). ``--task page`` (the
  default) a sample per page, asked ``OCR:`` and answered with the Markdown, as DeepSeek-OCR 2's Free OCR is;
  ``--task spotting`` a sample per page, asked ``Spotting:`` and answered with each line's text and box
  (:func:`diffu_page.page.labels.to_spotting`); ``--task elements`` a sample per layout element, for its recogniser:
  the element's crop, cut from its page into ``<out stem>_crops/`` as ``<page id>_<k>.png``, its recognition prompt
  (``OCR:`` or ``Table Recognition:``) and the text it reads there (a table as OTSL).
- ``unlimited-ocr-2``: as ms-swift fine-tunes Unlimited-OCR 2 (``messages`` and ``images``; its model type
  ``unlimited-ocr`` and template ``unlimited_ocr``, which shows each ``<image>`` as ``<PAGE><image>``, the prompt its
  own inference sends). ``--task parse`` (the default) a sample per page, asked ``<image>Multi page parsing.`` and
  answered with ``<PAGE>`` and a block per element (:func:`diffu_page.page.labels.to_unlimited_ocr`); ``--task
  opening`` a sample per scan, its pages its images in reading order (an opening's two:
  ``<image><image>Multi page parsing.``), answered with a ``<PAGE>`` each.
- ``minicpm-ocr``: as ms-swift fine-tunes MiniCPM-OCR 1.0 (model type and template ``minicpm_ocr``, which puts a line
  break after each image). ``--task elements`` (the default) a sample per element, its crop asked
  ``<image>Text Recognition:`` and answered with its lines, a table's asked ``<image>Table Recognition:`` and answered
  in OTSL between ``<otsl>`` and ``</otsl>``; ``--task page`` a sample per page, asked ``<image>Text Recognition:`` and
  answered with its plain text (:func:`diffu_page.page.labels.to_plain_text`).

A page that is the whole of its image is shown as that image; a page of an opening is cut from its scan into
``<out stem>_pages/<page id>.jpg``. With ``--xml``, ``--split-from-gt-test-lists`` sends every scan its volume's test
lists name to ``<out stem>.test.jsonl`` (and ``<out stem>.test.meta.jsonl``), never to ``--out``, and ``--exclude``
leaves out the pages a file lists (by id, as the MOSS page-pair sets give them, or by path: pages a generator was
trained on); the GT's second export of a collection is always left out.

    diffu-reader-set --run runs/pages --model deepseek-ocr-2 --out data/readers/deepseek_ocr_2.jsonl
    diffu-reader-set --xml ~/Desktop/data/GT-Lejonet_HTR_202602 --model unlimited-ocr-2 --split-from-gt-test-lists \\
        --exclude data/readers/generator_pages.txt --out data/readers/unlimited_ocr_2.jsonl
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from diffu_page.data.build_crops import TILE_QUALITY
from diffu_page.data.reader_pages import (
    ReaderPage,
    ReaderScan,
    SampleMeta,
    Skipped,
    SkipReason,
    Split,
    gt_split,
    open_scan,
    read_exclusions,
    run_scans,
    unsplit,
    xml_scans,
)
from diffu_page.local_run import describe
from diffu_page.output import console
from diffu_page.page.labels import (
    OCR_PROMPT,
    PARSE_PROMPT,
    SPOTTING_PROMPT,
    TEXT_PROMPT,
    PageElement,
    PageElements,
    Recognition,
    minicpm_ocr_recognition,
    paddleocr_vl_recognition,
    to_grounded_markdown,
    to_markdown,
    to_plain_text,
    to_spotting,
    to_unlimited_ocr,
)

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class Reader(StrEnum):
    """The page readers a set is written for."""

    DEEPSEEK_OCR_2 = "deepseek-ocr-2"
    PADDLEOCR_VL = "paddleocr-vl"
    UNLIMITED_OCR_2 = "unlimited-ocr-2"
    MINICPM_OCR = "minicpm-ocr"


class Task(StrEnum):
    """What a set teaches its reader: a whole page grounded (DeepSeek-OCR 2), as Markdown or plain text, its lines
    spotted (PaddleOCR-VL) or parsed into blocks (Unlimited-OCR 2), a scan's pages parsed together (Unlimited-OCR 2's
    opening), or each element's crop (a recogniser's)."""

    GROUNDED = "grounded"
    MARKDOWN = "markdown"
    PAGE = "page"
    SPOTTING = "spotting"
    ELEMENTS = "elements"
    PARSE = "parse"
    OPENING = "opening"


TASKS: dict[Reader, tuple[Task, ...]] = {  # the tasks each reader is fine-tuned on, its default first
    Reader.DEEPSEEK_OCR_2: (Task.GROUNDED, Task.MARKDOWN),
    Reader.PADDLEOCR_VL: (Task.PAGE, Task.SPOTTING, Task.ELEMENTS),
    Reader.UNLIMITED_OCR_2: (Task.PARSE, Task.OPENING),
    Reader.MINICPM_OCR: (Task.ELEMENTS, Task.PAGE),
}


class PageFormat(BaseModel):
    """A page task's sample: the prompt the page's image is shown with, and its answer written from the page."""

    model_config = {"frozen": True}

    prompt: str
    answer: Callable[[PageElements], str]


def _parsed(page: PageElements) -> str:
    """A page alone as Unlimited-OCR 2 parses it."""
    return to_unlimited_ocr([page])


# DeepSeek-OCR 2's prompts for a document and for one without layouts, without the trailing space its model card shows
# (its own inference strips a turn's content, so the model never reads one there); the others after the image's
# placeholder, as PaddleFormers' samples and ms-swift's write it
PAGE_FORMATS: dict[tuple[Reader, Task], PageFormat] = {
    (Reader.DEEPSEEK_OCR_2, Task.GROUNDED): PageFormat(
        prompt="<image>\n<|grounding|>Convert the document to markdown.", answer=to_grounded_markdown
    ),
    (Reader.DEEPSEEK_OCR_2, Task.MARKDOWN): PageFormat(prompt="<image>\nFree OCR.", answer=to_markdown),
    (Reader.PADDLEOCR_VL, Task.PAGE): PageFormat(prompt=f"<image>{OCR_PROMPT}", answer=to_markdown),
    (Reader.PADDLEOCR_VL, Task.SPOTTING): PageFormat(prompt=f"<image>{SPOTTING_PROMPT}", answer=to_spotting),
    (Reader.UNLIMITED_OCR_2, Task.PARSE): PageFormat(prompt=f"<image>{PARSE_PROMPT}", answer=_parsed),
    (Reader.MINICPM_OCR, Task.PAGE): PageFormat(prompt=f"<image>{TEXT_PROMPT}", answer=to_plain_text),
}

ELEMENT_FORMATS: dict[
    Reader, Callable[[PageElement], Recognition]
] = {  # an element's crop, as each recogniser reads it
    Reader.PADDLEOCR_VL: paddleocr_vl_recognition,
    Reader.MINICPM_OCR: minicpm_ocr_recognition,
}


class Turn(BaseModel):
    """One turn of a training conversation: who speaks, what, and (DeepSeek-OCR 2's user turn) the images it shows."""

    role: str
    content: str
    images: list[str] | None = None


class Sample(BaseModel):
    """One line of a set: its conversation, and (but DeepSeek-OCR 2's) the images its ``<image>`` placeholders show."""

    messages: list[Turn]
    images: list[str] | None = None


def deepseek_sample(prompt: str, answer: str, images: list[Path]) -> Sample:
    """A sample as DeepSeek-OCR 2 is fine-tuned on it: asked ``prompt`` in a turn that shows ``images``, answered."""
    return Sample(
        messages=[
            Turn(role="<|User|>", content=prompt, images=[str(image) for image in images]),
            Turn(role="<|Assistant|>", content=answer),
        ]
    )


def chat_sample(prompt: str, answer: str, images: list[Path]) -> Sample:
    """A sample as PaddleFormers and ms-swift fine-tune a reader: a user turn asking ``prompt`` and the assistant's
    answer, with the images its placeholders show, in their order."""
    return Sample(
        messages=[Turn(role="user", content=prompt), Turn(role="assistant", content=answer)],
        images=[str(image) for image in images],
    )


SAMPLE_SHAPES: dict[Reader, Callable[[str, str, list[Path]], Sample]] = {
    Reader.DEEPSEEK_OCR_2: deepseek_sample,
    Reader.PADDLEOCR_VL: chat_sample,
    Reader.UNLIMITED_OCR_2: chat_sample,
    Reader.MINICPM_OCR: chat_sample,
}


class ImageDirs(BaseModel):
    """Where a set's cut images go: pages cut from their scans, and elements' crops."""

    pages: Path
    crops: Path


def page_image(page: ReaderPage, pages: Path) -> Path:
    """The image showing ``page`` alone: its own when the page is all of it, else the page cut from it into ``pages``
    as ``<page id>.jpg``."""
    if page.rect is None:
        return page.image
    path = pages / f"{page.meta.id}.jpg"
    pages.mkdir(parents=True, exist_ok=True)
    with open_scan(page.image) as scan:
        scan.crop(page.rect).convert("RGB").save(path, quality=TILE_QUALITY)
    return path


def cut_elements(page: ReaderPage, crops: Path) -> list[Path]:
    """Each element of ``page`` cut from its image at its box (where the page lies on the image) and saved in ``crops``
    as ``<page id>_<k>.png`` (``k`` its place in reading order): their paths."""
    left, top = page.rect[:2] if page.rect is not None else (0, 0)
    paths: list[Path] = []
    with open_scan(page.image) as image:
        for k, element in enumerate(page.elements.elements):
            x0, y0, x1, y1 = element.box
            path = crops / f"{page.meta.id}_{k}.png"
            image.crop((x0 + left, y0 + top, x1 + left, y1 + top)).convert("RGB").save(path)
            paths.append(path)
    return paths


def element_samples(page: ReaderPage, model: Reader, crops: Path) -> list[tuple[Sample, SampleMeta]]:
    """Each element of ``page`` as ``model``'s recogniser is fine-tuned on it: its crop (:func:`cut_elements`) asked with
    its recognition prompt and answered with what it reads there; its meta the page's, its id ``<page id>_<k>``."""
    samples: list[tuple[Sample, SampleMeta]] = []
    for k, (element, crop) in enumerate(zip(page.elements.elements, cut_elements(page, crops), strict=True)):
        recognition = ELEMENT_FORMATS[model](element)
        sample = chat_sample(f"<image>{recognition.prompt}", recognition.answer, [crop])
        samples.append((sample, page.meta.model_copy(update={"id": f"{page.meta.id}_{k}"})))
    return samples


def opening_sample(scan: ReaderScan, pages: Path) -> tuple[Sample, SampleMeta]:
    """A scan as Unlimited-OCR 2 parses a document: its pages' images in reading order (an opening's two), asked
    ``<image><image>Multi page parsing.``, answered with a ``<PAGE>`` each; its meta its page's, or for two pages the
    left one's, its id the scan's and its side None."""
    images = [page_image(page, pages) for page in scan.pages]
    answer = to_unlimited_ocr([page.elements for page in scan.pages])
    first = scan.pages[0].meta
    meta = first if len(scan.pages) == 1 else first.model_copy(update={"id": scan.id, "side": None})
    return chat_sample("<image>" * len(images) + PARSE_PROMPT, answer, images), meta


def scan_samples(scan: ReaderScan, *, model: Reader, task: Task, dirs: ImageDirs) -> list[tuple[Sample, SampleMeta]]:
    """A scan's samples for ``model``'s ``task``, each with its meta: its pages in one (``opening``), a sample per
    element of each page (``elements``), else a sample per page."""
    match task:
        case Task.OPENING:
            return [opening_sample(scan, dirs.pages)]
        case Task.ELEMENTS:
            return [sample for page in scan.pages for sample in element_samples(page, model, dirs.crops)]
        case _:
            page_format, shape = PAGE_FORMATS[(model, task)], SAMPLE_SHAPES[model]
            return [
                (
                    shape(page_format.prompt, page_format.answer(page.elements), [page_image(page, dirs.pages)]),
                    page.meta,
                )
                for page in scan.pages
            ]


def set_files(out: Path, split: Split | None) -> tuple[Path, Path]:
    """Where a split's samples and their meta go: ``out`` and ``<out stem>.meta.jsonl`` beside it, a test split's to
    ``<out stem>.test.jsonl`` and ``<out stem>.test.meta.jsonl``."""
    if split is Split.TEST:
        return out.with_name(f"{out.stem}.test.jsonl"), out.with_name(f"{out.stem}.test.meta.jsonl")
    return out, out.with_name(f"{out.stem}.meta.jsonl")


type ScanSamples = tuple[ReaderScan, list[tuple[Sample, SampleMeta]]]  # a scan and its samples, each with its meta


def scans_written(
    items: Iterable[ReaderScan | Skipped], *, model: Reader, task: Task, dirs: ImageDirs
) -> Iterator[ScanSamples | Skipped]:
    """Each scan of ``items`` with its samples (:func:`scan_samples`), its images written, and each item left out: one
    whose image cannot be read whole (a torn file whose size still reads) is left out too, not the set."""
    for item in items:
        match item:
            case Skipped():
                yield item
            case ReaderScan():
                try:
                    samples = scan_samples(item, model=model, task=task, dirs=dirs)
                except OSError as error:
                    yield Skipped(labels=item.labels, reason=SkipReason.UNREADABLE, error=describe(error))
                    continue
                yield item, samples


class Written(BaseModel):
    """What writing a set came to: its samples by split (None: not split), the pages and scans they show, and the
    files, scans or pages left out, by why."""

    samples: dict[Split | None, int]
    pages: int
    scans: int
    skipped: dict[str, int]


def write_set(
    items: Iterable[ReaderScan | Skipped], out: Path, *, model: Reader, task: Task, splits: tuple[Split | None, ...]
) -> Written:
    """Each scan of ``items`` as ``model``'s ``task`` samples, written to its split's set files (:func:`set_files`, one
    pair for each of ``splits``) as it is taken, its images beside them; and each item left out, in
    ``<out stem>.skipped.jsonl``."""
    dirs = ImageDirs(
        pages=(out.parent / f"{out.stem}_pages").resolve(), crops=(out.parent / f"{out.stem}_crops").resolve()
    )
    if task is Task.ELEMENTS:
        dirs.crops.mkdir(parents=True, exist_ok=True)
    samples: Counter[Split | None] = Counter()
    skipped: Counter[str] = Counter()
    pages = scans = 0
    with ExitStack() as stack:
        files = {
            split: [stack.enter_context(path.open("w", encoding="utf-8")) for path in set_files(out, split)]
            for split in splits
        }
        left_out = stack.enter_context(out.with_name(f"{out.stem}.skipped.jsonl").open("w", encoding="utf-8"))
        for outcome in scans_written(items, model=model, task=task, dirs=dirs):
            match outcome:
                case Skipped():
                    left_out.write(f"{outcome.model_dump_json()}\n")
                    skipped[outcome.reason] += 1
                case (scan, scan_lines):
                    for sample, meta in scan_lines:
                        set_file, meta_file = files[meta.split]
                        set_file.write(f"{sample.model_dump_json(exclude_none=True)}\n")
                        meta_file.write(f"{meta.model_dump_json()}\n")
                        samples[meta.split] += 1
                    pages += len(scan.pages)
                    scans += 1
    return Written(samples=dict(samples), pages=pages, scans=scans, skipped=dict(skipped))


def _run_items(run: Path) -> list[ReaderScan]:
    """A run's images as scans of one page each (:func:`run_scans`).

    Raises:
        typer.BadParameter: If the run holds no elements files, or one names an image the run does not hold.
    """
    try:
        return run_scans(run)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--run") from error


def _xml_items(
    xml: Path, *, split_of: Callable[[Path], Split | None], excluded: frozenset[str]
) -> Iterable[ReaderScan | Skipped]:
    """The scans of the PAGE or ALTO files ``xml`` gives, and what of them is left out (:func:`xml_scans`).

    Raises:
        typer.BadParameter: If ``xml`` gives no PAGE or ALTO file.
    """
    try:
        return xml_scans(xml, split_of=split_of, excluded=excluded)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--xml") from error


@app.command()
def reader_set(
    model: Annotated[Reader, typer.Option(help="the page reader to fine-tune")],
    out: Annotated[
        Path,
        typer.Option(
            help="the JSON-lines file to write; its meta, test split, left-out record, cut pages (<out stem>_pages) "
            "and crops (<out stem>_crops) go beside it"
        ),
    ],
    run: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False, help="a diffu-generate run drawn with --reader-labels")
    ] = None,
    xml: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            help="PAGE or ALTO files with their scans: a folder of them (the GT's root, a diffu-run-export's folder) "
            "or a file listing them, one a line",
        ),
    ] = None,
    task: Annotated[
        Task | None,
        typer.Option(
            help="deepseek-ocr-2: grounded (the grounded Markdown; default) or markdown (Free OCR: the Markdown alone); "
            "paddleocr-vl: page (OCR: the Markdown; default), spotting (each line's text and box) or elements (each "
            "element's crop, for its recogniser); unlimited-ocr-2: parse (a page's blocks; default) or opening (a "
            "scan's pages in one sample); minicpm-ocr: elements (each element's crop; default) or page (its plain text)"
        ),
    ] = None,
    split_from_gt_test_lists: Annotated[
        bool,
        typer.Option(
            help="--xml: a scan a test list of its volume names (the GT's test_*) goes to <out stem>.test.jsonl, "
            "never to --out"
        ),
    ] = False,
    exclude: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="--xml: a file listing pages to leave out, one a line: a page's id (a MOSS page-pair set's) or the "
            "path of its PAGE file or scan, whole or its last parts",
        ),
    ] = None,
) -> None:
    """Write the pages of a run, or of PAGE or ALTO files, as a page reader's fine-tuning set."""
    task = task or TASKS[model][0]
    if task not in TASKS[model]:
        raise typer.BadParameter(f"{model} is fine-tuned on {', '.join(TASKS[model])}", param_hint="--task")
    if run is not None and xml is None:
        if split_from_gt_test_lists or exclude is not None:
            raise typer.BadParameter(
                "a run's pages are drawn, neither split by the GT's test lists nor left out by a list: give --xml",
                param_hint="--split-from-gt-test-lists, --exclude",
            )
        items = _run_items(run)
    elif xml is not None and run is None:
        excluded = read_exclusions(exclude) if exclude is not None else frozenset[str]()
        items = _xml_items(xml, split_of=gt_split if split_from_gt_test_lists else unsplit, excluded=excluded)
    else:
        raise typer.BadParameter("give the pages as --run or as --xml, one of them", param_hint="--run, --xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    splits = (Split.TRAIN, Split.TEST) if split_from_gt_test_lists else (None,)
    written = write_set(items, out, model=model, task=task, splits=splits)
    for split in splits:
        console.print(f"{written.samples.get(split, 0)} samples -> {set_files(out, split)[0]}")
    console.print(f"from {written.pages} pages of {written.scans} scans")
    if written.skipped:
        reasons = ", ".join(f"{count} {reason}" for reason, count in sorted(written.skipped.items()))
        console.print(f"left out: {reasons} ({out.with_name(f'{out.stem}.skipped.jsonl')})")


if __name__ == "__main__":
    app()
