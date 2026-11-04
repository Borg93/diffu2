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
  in OTSL between ``<otsl>`` and ``</otsl>``; ``--task lines`` a sample per text line of every element (a table
  cell's too), its box cut from its page into ``<out stem>_lines/`` as ``<page id>_<k>_<j>.jpg`` (element ``k``, line
  ``j``), asked ``<image>Text Recognition:`` and answered with the line's text, as OpenBMB fine-tunes it on
  handwritten lines; ``--task page`` a sample per page, asked ``<image>Text Recognition:`` and answered with its plain
  text (:func:`diffu_page.page.labels.to_plain_text`). A line with no text, a drawn line tagged ``unclear`` and a box
  narrower or lower than :data:`MIN_LINE_SIDE` px are left out and recorded. ``--max-crop-pixels`` scales an element's
  or a line's crop down to at most that many pixels, its aspect kept (200,704 = 448 x 448 keeps a MiniCPM-OCR line one
  view of 256 visual tokens); the meta of a crop's sample also gives its element's kind and place, a line's its
  production and place in its element.

A page that is the whole of its image is shown as that image; a page of an opening is cut from its scan into
``<out stem>_pages/<page id>.jpg``. With ``--xml``, ``--split-from-gt-test-lists`` sends every scan its volume's test
lists name to ``<out stem>.test.jsonl`` (and ``<out stem>.test.meta.jsonl``), never to ``--out``, and ``--exclude``
leaves out the pages a file lists (by id, as the MOSS page-pair sets give them, or by path: pages a generator was
trained on); the GT's second export of a collection is always left out.

A set is one variant of a reader's data, so several can be built and trained on apart: ``--collection`` takes only the
collections it names, ``--exclude-collection`` leaves some out, ``--max-per-collection`` caps a collection's training
samples, never its test ones (``--seed`` shuffles the scans first, so a cap takes from all of a collection's volumes),
and ``--max-crop-pixels`` scales its crops; real and drawn pages are two sets (``--xml`` of the GT, of a
``diffu-run-export``), mixed when a run trains on both. ``--gt-test-list`` takes one of a volume's nested test lists
(``test_0050``) for the test side instead of all of them; ``--mask-lines`` whitens a line outside its outline,
``--min-line-side`` raises the smallest line kept and ``--reject-uncertain`` leaves out a line with a transcriber's
doubt or comment, as a TrOCR baseline trained on the GT cuts and keeps its lines. ``--workers`` cuts the scans in
that many processes, the set the same. ``<out stem>.options.json`` records every option and what the set came to.

    diffu-reader-set --run runs/pages --model deepseek-ocr-2 --out data/readers/deepseek_ocr_2.jsonl
    diffu-reader-set --xml ~/Desktop/data/GT-Lejonet_HTR_202602 --model unlimited-ocr-2 --split-from-gt-test-lists \\
        --exclude data/readers/generator_pages.txt --out data/readers/unlimited_ocr_2.jsonl
    diffu-reader-set --xml ~/Desktop/data/GT --model minicpm-ocr --task lines --max-crop-pixels 200704 \\
        --split-from-gt-test-lists --out data/readers/minicpm_lines.jsonl
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, TextIO

import typer
from PIL import Image, ImageDraw
from pydantic import BaseModel

from diffu_page.data.build_crops import TILE_QUALITY
from diffu_page.data.reader_pages import (
    TEST_LISTS,
    ReaderPage,
    ReaderScan,
    SampleMeta,
    Selection,
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
    ElementLine,
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
    opening), or each element's or line's crop (a recogniser's)."""

    GROUNDED = "grounded"
    MARKDOWN = "markdown"
    PAGE = "page"
    SPOTTING = "spotting"
    ELEMENTS = "elements"
    LINES = "lines"
    PARSE = "parse"
    OPENING = "opening"


TASKS: dict[Reader, tuple[Task, ...]] = {  # the tasks each reader is fine-tuned on, its default first
    Reader.DEEPSEEK_OCR_2: (Task.GROUNDED, Task.MARKDOWN),
    Reader.PADDLEOCR_VL: (Task.PAGE, Task.SPOTTING, Task.ELEMENTS),
    Reader.UNLIMITED_OCR_2: (Task.PARSE, Task.OPENING),
    Reader.MINICPM_OCR: (Task.ELEMENTS, Task.LINES, Task.PAGE),
}
MIN_LINE_SIDE = 8  # px: a line's box narrower or lower than this holds no writing (a transcriber's stray box)
UNCERTAIN_MARKS = ("??", "(?)", "[", "]")  # a transcriber's doubt or comment in a line's text (the TrOCR baseline's)


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
    """Where a set's cut images go: pages cut from their scans, elements' crops and lines' crops."""

    pages: Path
    crops: Path
    lines: Path


def fitted(image: Image.Image, max_pixels: int | None) -> Image.Image:
    """``image`` scaled down, its aspect kept, to at most ``max_pixels`` pixels (each side at least 1 px); as it is
    when it holds no more, or when ``max_pixels`` is None."""
    if max_pixels is None or image.width * image.height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (image.width * image.height))
    size = (max(1, math.floor(image.width * scale)), max(1, math.floor(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


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


def _on_image(box: tuple[int, int, int, int], page: ReaderPage) -> tuple[int, int, int, int]:
    """``box`` (the page's px) on the image the page is on."""
    left, top = page.rect[:2] if page.rect is not None else (0, 0)
    x0, y0, x1, y1 = box
    return x0 + left, y0 + top, x1 + left, y1 + top


def cut_elements(page: ReaderPage, crops: Path, max_pixels: int | None = None) -> list[Path]:
    """Each element of ``page`` cut from its image at its box (where the page lies on the image), scaled to at most
    ``max_pixels`` (:func:`fitted`) and saved in ``crops`` as ``<page id>_<k>.png`` (``k`` its place in reading order):
    their paths."""
    paths: list[Path] = []
    with open_scan(page.image) as image:
        for k, element in enumerate(page.elements.elements):
            path = crops / f"{page.meta.id}_{k}.png"
            fitted(image.crop(_on_image(element.box, page)).convert("RGB"), max_pixels).save(path)
            paths.append(path)
    return paths


def element_samples(
    page: ReaderPage, model: Reader, *, crops: Path, max_pixels: int | None = None
) -> list[tuple[Sample, SampleMeta]]:
    """Each element of ``page`` as ``model``'s recogniser is fine-tuned on it: its crop (:func:`cut_elements`) asked with
    its recognition prompt and answered with what it reads there; its meta the page's, its id ``<page id>_<k>``, with
    the element's kind and place."""
    samples: list[tuple[Sample, SampleMeta]] = []
    cut = cut_elements(page, crops, max_pixels)
    for k, (element, crop) in enumerate(zip(page.elements.elements, cut, strict=True)):
        recognition = ELEMENT_FORMATS[model](element)
        sample = chat_sample(f"<image>{recognition.prompt}", recognition.answer, [crop])
        meta = page.meta.model_copy(update={"id": f"{page.meta.id}_{k}", "kind": element.kind, "element": k})
        samples.append((sample, meta))
    return samples


class LineRules(BaseModel):
    """How a line is cut and which lines are kept: masked to its outline (white outside it, as the GT's TrOCR baseline
    cuts its lines and its out-of-domain test lines are cut) or its box as it is; its box's shorter side at least
    ``min_side`` px; and, with ``reject_uncertain``, a line whose text holds a transcriber's doubt or comment
    (:data:`UNCERTAIN_MARKS`) left out, as the baseline leaves it out."""

    model_config = {"frozen": True}

    mask: bool = False
    min_side: int = MIN_LINE_SIDE
    reject_uncertain: bool = False


def _line_left_out(line: ElementLine, rules: LineRules) -> SkipReason | None:
    """Why a line is no sample under ``rules``: no text, tagged unclear (a drawn line that read back badly), a box too
    small to hold writing, or a transcriber's doubt or comment in its text; None for a line to read."""
    if not line.text.strip():
        return SkipReason.EMPTY_LINE
    if line.tag == "unclear":
        return SkipReason.UNCLEAR_LINE
    x0, y0, x1, y1 = line.box
    if min(x1 - x0, y1 - y0) < rules.min_side:
        return SkipReason.SMALL_LINE
    if rules.reject_uncertain and any(mark in line.text for mark in UNCERTAIN_MARKS):
        return SkipReason.UNCERTAIN_LINE
    return None


def masked(crop: Image.Image, polygon: list[tuple[int, int]]) -> Image.Image:
    """``crop`` white outside ``polygon`` (in the crop's px), as the GT's TrOCR baseline masks a line to its outline;
    as it is for an outline of fewer than three points."""
    if len(polygon) < 3:
        return crop
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    return Image.composite(crop, Image.new(crop.mode, crop.size, (255, 255, 255)), mask)


def line_samples(
    page: ReaderPage, labels: str, *, lines: Path, max_pixels: int | None = None, rules: LineRules | None = None
) -> list[tuple[Sample, SampleMeta] | Skipped]:
    """Each text line of each element of ``page`` (a table cell's too) as MiniCPM-OCR is fine-tuned on a line: its box
    cut from the page's image, scaled to at most ``max_pixels`` (:func:`fitted`) and saved in ``lines`` as
    ``<page id>_<k>_<j>.jpg`` (element ``k``, line ``j``), asked ``<image>Text Recognition:`` and answered with its
    text; its meta the page's, its id the crop's stem, with its element's kind and place, its production and its place
    in the element. With ``rules.mask``, a line is white outside its outline (:func:`masked`). A line that is no sample
    (:func:`_line_left_out`) is left out, recorded with ``labels`` (its page's label file)."""
    rules = rules or LineRules()
    samples: list[tuple[Sample, SampleMeta] | Skipped] = []
    with open_scan(page.image) as image:
        for k, element in enumerate(page.elements.elements):
            for j, line in enumerate(element.lines):
                line_id = f"{page.meta.id}_{k}_{j}"
                reason = _line_left_out(line, rules)
                if reason is not None:
                    samples.append(Skipped(labels=labels, side=page.meta.side, reason=reason, id=line_id))
                    continue
                path = lines / f"{line_id}.jpg"
                crop = image.crop(_on_image(line.box, page)).convert("RGB")
                if rules.mask:
                    x0, y0 = line.box[:2]
                    crop = masked(crop, [(x - x0, y - y0) for x, y in line.polygon])
                crop = fitted(crop, max_pixels)
                crop.save(path, quality=TILE_QUALITY)
                update = {"id": line_id, "kind": element.kind, "production": line.production, "element": k, "line": j}
                sample = chat_sample(f"<image>{TEXT_PROMPT}", line.text, [path])
                samples.append((sample, page.meta.model_copy(update=update)))
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


type SampleOrSkipped = tuple[Sample, SampleMeta] | Skipped  # a sample with its meta, or a line left out


def scan_samples(
    scan: ReaderScan,
    *,
    model: Reader,
    task: Task,
    dirs: ImageDirs,
    max_crop_pixels: int | None = None,
    rules: LineRules | None = None,
) -> list[SampleOrSkipped]:
    """A scan's samples for ``model``'s ``task``, each with its meta, and the lines left out: its pages in one
    (``opening``), a sample per element of each page (``elements``) or per line (``lines``), else a sample per page;
    a crop scaled to at most ``max_crop_pixels``, a line cut and kept by ``rules``."""
    match task:
        case Task.OPENING:
            return [opening_sample(scan, dirs.pages)]
        case Task.ELEMENTS:
            return [
                sample
                for page in scan.pages
                for sample in element_samples(page, model, crops=dirs.crops, max_pixels=max_crop_pixels)
            ]
        case Task.LINES:
            return [
                sample
                for page in scan.pages
                for sample in line_samples(page, scan.labels, lines=dirs.lines, max_pixels=max_crop_pixels, rules=rules)
            ]
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


type ScanSamples = tuple[ReaderScan, list[SampleOrSkipped]]  # a scan, its samples with their meta, its lines left out


class Limits(BaseModel):
    """How much of its input a set takes: a crop's pixels at most (None: as cut), a collection's training (or
    unsplit) samples (None: all), a collection taking no more scans once it holds that many (its last scan's are all
    kept; the test side is never capped, so every variant of a set is tested on the same pages), and how its lines are
    cut and kept (:class:`LineRules`)."""

    model_config = {"frozen": True}

    max_crop_pixels: int | None = None
    max_per_collection: int | None = None
    line_rules: LineRules = LineRules()


def _scan_written(
    item: ReaderScan | Skipped,
    *,
    model: Reader,
    task: Task,
    dirs: ImageDirs,
    max_crop_pixels: int | None,
    rules: LineRules,
) -> ScanSamples | Skipped:
    """``item`` with its samples (:func:`scan_samples`), its images written; an item left out as it is, and a scan
    whose image cannot be read whole (a torn file whose size still reads) left out too, not the set."""
    if isinstance(item, Skipped):
        return item
    try:
        samples = scan_samples(item, model=model, task=task, dirs=dirs, max_crop_pixels=max_crop_pixels, rules=rules)
        return item, samples
    except OSError as error:
        return Skipped(labels=item.labels, reason=SkipReason.UNREADABLE, error=describe(error))


def scans_written(
    items: Iterable[ReaderScan | Skipped],
    *,
    model: Reader,
    task: Task,
    dirs: ImageDirs,
    limits: Limits | None = None,
    workers: int = 1,
) -> Iterator[ScanSamples | Skipped]:
    """Each scan of ``items`` with its samples, its images written, and each item left out (:func:`_scan_written`), in
    the order of ``items``: one whose collection already holds ``limits.max_per_collection`` samples of its split is
    left out too. With more than one of ``workers`` and no cap, the scans are cut in that many processes at once (a
    cap decides each scan by the ones before it, so a capped set is cut in one)."""
    limits = limits or Limits()
    cut = partial(
        _scan_written,
        model=model,
        task=task,
        dirs=dirs,
        max_crop_pixels=limits.max_crop_pixels,
        rules=limits.line_rules,
    )
    if workers > 1 and limits.max_per_collection is None:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(cut, items, chunksize=4)
        return
    held: Counter[tuple[str, Split | None]] = Counter()
    for item in items:
        if isinstance(item, ReaderScan):
            first = item.pages[0].meta
            if (
                limits.max_per_collection is not None
                and first.split is not Split.TEST  # the test side is never capped: variants share their test set
                and held[first.collection, first.split] >= limits.max_per_collection
            ):
                yield Skipped(labels=item.labels, reason=SkipReason.CAPPED)
                continue
        outcome = cut(item)
        if not isinstance(outcome, Skipped):
            scan, samples = outcome
            first = scan.pages[0].meta
            held[first.collection, first.split] += sum(1 for sample in samples if not isinstance(sample, Skipped))
        yield outcome


class Written(BaseModel):
    """What writing a set came to: its samples by split (None: not split), the pages and scans they show, and the
    files, scans or pages left out, by why."""

    samples: dict[Split | None, int]
    pages: int
    scans: int
    skipped: dict[str, int]


class _SetWriter:
    """A set's open files as it is written, and what went where: each split's set and meta files (:func:`set_files`),
    the record of what is left out, the samples by split and the left out by why."""

    def __init__(self, files: dict[Split | None, tuple[TextIO, TextIO]], left_out: TextIO) -> None:
        self._files = files
        self._left_out = left_out
        self.samples: Counter[Split | None] = Counter()
        self.skipped: Counter[str] = Counter()

    def write(self, entry: SampleOrSkipped) -> None:
        """A sample and its meta written to its split's files, or an item or line left out to the record."""
        match entry:
            case Skipped():
                self._left_out.write(f"{entry.model_dump_json()}\n")
                self.skipped[entry.reason] += 1
            case (sample, meta):
                set_file, meta_file = self._files[meta.split]
                set_file.write(f"{sample.model_dump_json(exclude_none=True)}\n")
                meta_file.write(f"{meta.model_dump_json()}\n")
                self.samples[meta.split] += 1


def write_set(
    items: Iterable[ReaderScan | Skipped],
    out: Path,
    *,
    model: Reader,
    task: Task,
    splits: tuple[Split | None, ...],
    limits: Limits | None = None,
    workers: int = 1,
) -> Written:
    """Each scan of ``items`` as ``model``'s ``task`` samples, written to its split's set files (:func:`set_files`, one
    pair for each of ``splits``) as it is taken, its images beside them, within ``limits``; and each item or line left
    out, in ``<out stem>.skipped.jsonl``; the scans cut by ``workers`` processes (:func:`scans_written`)."""
    dirs = ImageDirs(
        pages=(out.parent / f"{out.stem}_pages").resolve(),
        crops=(out.parent / f"{out.stem}_crops").resolve(),
        lines=(out.parent / f"{out.stem}_lines").resolve(),
    )
    crop_dirs = {Task.ELEMENTS: dirs.crops, Task.LINES: dirs.lines}
    if task in crop_dirs:
        crop_dirs[task].mkdir(parents=True, exist_ok=True)
    pages = scans = 0
    with ExitStack() as stack:
        files: dict[Split | None, tuple[TextIO, TextIO]] = {}
        for split in splits:
            set_path, meta_path = set_files(out, split)
            files[split] = (
                stack.enter_context(set_path.open("w", encoding="utf-8")),
                stack.enter_context(meta_path.open("w", encoding="utf-8")),
            )
        left_out = stack.enter_context(out.with_name(f"{out.stem}.skipped.jsonl").open("w", encoding="utf-8"))
        writer = _SetWriter(files, left_out)
        for outcome in scans_written(items, model=model, task=task, dirs=dirs, limits=limits, workers=workers):
            match outcome:
                case Skipped():
                    writer.write(outcome)
                case (scan, entries):
                    for entry in entries:
                        writer.write(entry)
                    pages += len(scan.pages)
                    scans += 1
    return Written(samples=dict(writer.samples), pages=pages, scans=scans, skipped=dict(writer.skipped))


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
    xml: Path, *, split_of: Callable[[Path], Split | None], excluded: frozenset[str], selection: Selection
) -> Iterable[ReaderScan | Skipped]:
    """The scans of the PAGE or ALTO files ``xml`` gives, and what of them is left out (:func:`xml_scans`).

    Raises:
        typer.BadParameter: If ``xml`` gives no PAGE or ALTO file.
    """
    try:
        return xml_scans(xml, split_of=split_of, excluded=excluded, selection=selection)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--xml") from error


class SetOptions(BaseModel):
    """``<out stem>.options.json``: what a set was built from and with, so the variants of one reader's sets are told
    apart and a run's manifest can name them; and what it came to."""

    model: Reader
    task: Task
    run: str | None
    xml: str | None
    split_from_gt_test_lists: bool
    gt_test_lists: str | None = None  # which of a volume's lists name its test pages (all: test_*)
    exclude: str | None
    selection: Selection
    limits: Limits
    samples: dict[str, int]  # by split ("all": the set is not split)
    pages: int
    scans: int
    skipped: dict[str, int]


@app.command()
def reader_set(
    model: Annotated[Reader, typer.Option(help="the page reader to fine-tune")],
    out: Annotated[
        Path,
        typer.Option(
            help="the JSON-lines file to write; its meta, test split, left-out record, cut pages (<out stem>_pages), "
            "crops (<out stem>_crops) and line crops (<out stem>_lines) go beside it"
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
            "scan's pages in one sample); minicpm-ocr: elements (each element's crop; default), lines (each text "
            "line's crop, in <out stem>_lines) or page (its plain text)"
        ),
    ] = None,
    max_crop_pixels: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="elements, lines: scale each crop down to at most this many pixels, its aspect kept (200704 = 448 x "
            "448: a MiniCPM-OCR line in one view)",
        ),
    ] = None,
    split_from_gt_test_lists: Annotated[
        bool,
        typer.Option(
            help="--xml: a scan a test list of its volume names (the GT's test_*) goes to <out stem>.test.jsonl, "
            "never to --out"
        ),
    ] = False,
    gt_test_list: Annotated[
        str,
        typer.Option(
            help="--split-from-gt-test-lists: which of a volume's test lists name its test pages, a file name or a "
            "pattern (test_0050: the 5 %% the GT's TrOCR baseline was tested on; by default all of them, test_*, "
            "the largest: 10 %%)"
        ),
    ] = TEST_LISTS,
    exclude: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="--xml: a file listing pages to leave out, one a line: a page's id (a MOSS page-pair set's) or the "
            "path of its PAGE file or scan, whole or its last parts",
        ),
    ] = None,
    collection: Annotated[
        list[str] | None,
        typer.Option(help="--xml: take only this collection (GT-...); give it again for more. None given: all"),
    ] = None,
    exclude_collection: Annotated[
        list[str] | None, typer.Option(help="--xml: leave this collection out; give it again for more")
    ] = None,
    max_per_collection: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="take no more training scans of a collection once it holds this many samples (with --seed, from all "
            "its volumes); the test split is never capped",
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="--xml: read the scans in an order shuffled by this seed, not by path")
    ] = None,
    mask_lines: Annotated[
        bool,
        typer.Option(
            help="lines: white outside each line's outline, as the GT's TrOCR baseline cuts its lines (default: the "
            "line's box as it is)"
        ),
    ] = False,
    min_line_side: Annotated[
        int, typer.Option(min=1, help="lines: leave out a line whose box is narrower or lower than this (px)")
    ] = MIN_LINE_SIDE,
    reject_uncertain: Annotated[
        bool,
        typer.Option(
            help="lines: leave out a line whose text holds a transcriber's doubt or comment (??, (?), [ or ]), as the "
            "GT's TrOCR baseline does"
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            min=1,
            help="cut the scans in this many processes at once (the set the same, in the same order); a set capped "
            "by --max-per-collection is cut in one",
        ),
    ] = 1,
) -> None:
    """Write the pages of a run, or of PAGE or ALTO files, as a page reader's fine-tuning set."""
    task = task or TASKS[model][0]
    if task not in TASKS[model]:
        raise typer.BadParameter(f"{model} is fine-tuned on {', '.join(TASKS[model])}", param_hint="--task")
    if max_crop_pixels is not None and task not in (Task.ELEMENTS, Task.LINES):
        raise typer.BadParameter(
            f"{task} cuts no crops: give it with elements or lines", param_hint="--max-crop-pixels"
        )
    selection = Selection(
        include=tuple(sorted(set(collection or ()))), exclude=tuple(sorted(set(exclude_collection or ()))), seed=seed
    )
    if run is not None and xml is None:
        if split_from_gt_test_lists or exclude is not None or selection != Selection():
            raise typer.BadParameter(
                "a run's pages are drawn, neither split by the GT's test lists, left out by a list nor of collections: "
                "give --xml",
                param_hint="--split-from-gt-test-lists, --exclude, --collection, --exclude-collection, --seed",
            )
        items = _run_items(run)
    elif xml is not None and run is None:
        excluded = read_exclusions(exclude) if exclude is not None else frozenset[str]()
        split_of = partial(gt_split, lists=gt_test_list) if split_from_gt_test_lists else unsplit
        items = _xml_items(xml, split_of=split_of, excluded=excluded, selection=selection)
    else:
        raise typer.BadParameter("give the pages as --run or as --xml, one of them", param_hint="--run, --xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    splits = (Split.TRAIN, Split.TEST) if split_from_gt_test_lists else (None,)
    rules = LineRules(mask=mask_lines, min_side=min_line_side, reject_uncertain=reject_uncertain)
    limits = Limits(max_crop_pixels=max_crop_pixels, max_per_collection=max_per_collection, line_rules=rules)
    written = write_set(items, out, model=model, task=task, splits=splits, limits=limits, workers=workers)
    options = SetOptions(
        model=model,
        task=task,
        run=str(run.resolve()) if run else None,
        xml=str(xml.resolve()) if xml else None,
        split_from_gt_test_lists=split_from_gt_test_lists,
        gt_test_lists=gt_test_list if split_from_gt_test_lists else None,
        exclude=str(exclude.resolve()) if exclude else None,
        selection=selection,
        limits=limits,
        samples={str(split or "all"): count for split, count in written.samples.items()},
        pages=written.pages,
        scans=written.scans,
        skipped=written.skipped,
    )
    out.with_name(f"{out.stem}.options.json").write_text(options.model_dump_json(indent=1) + "\n", encoding="utf-8")
    for split in splits:
        console.print(f"{written.samples.get(split, 0)} samples -> {set_files(out, split)[0]}")
    console.print(f"from {written.pages} pages of {written.scans} scans")
    if written.skipped:
        reasons = ", ".join(f"{count} {reason}" for reason, count in sorted(written.skipped.items()))
        console.print(f"left out: {reasons} ({out.with_name(f'{out.stem}.skipped.jsonl')})")


if __name__ == "__main__":
    app()
