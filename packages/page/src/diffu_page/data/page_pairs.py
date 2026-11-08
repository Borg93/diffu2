"""Page pairs for MOSS-Image's reference training: whole single pages of the handwritten ground truth, each with a
sample page of its own volume, and windows of them beside if asked.

Every PAGE file of the GT (``<root>/<collection>/<volume>/<volume>/page/<file>.xml``, macOS ``._*`` files passed over)
is read with its scan (``find_scan``) and split into its single pages (``scan_pages``: an opening's two, else the scan
whole). A scan wider than tall that its lines cannot split (``is_unsplit_spread``: an opening whose pages' lines
overlap side to side, or a single wide sheet, which the lines cannot tell apart) gives no page: it is skipped and
counted, as no spread is ever trained on. Each page is cut from its scan at about ``--megapixels`` on the latent grid,
exactly as ``diffu-generate --whole-page`` plans a page (``plan_whole_page``), and prompted as drawing prompts it: the
caption of its whole-page plan compiled as MOSS's Layout prompt (``diffu_moss.sample_caption.sample_layout_prompt``,
as ``MossLayoutPromptStep`` compiles it), every line's exact text in its box. The page is captioned in its own style:
the archive and the volume's years (``provenance_of``: whose hand it is, by name, as a v2 caption names its archive)
and the look measured on the page (``scan_look``: greyscale or colour, the paper's and the ink's colour). Each row
carries the page's prompt in all six forms the training's mixes draw from (``diffu_moss.train.pairs.Variant``;
``diffu-moss-cache-pairs`` fixes each pair's by its seed):

- no reference: the page's own caption (``plan_whole_page`` without a reference), hand and look in words;
- look: the sample image, and hand and look in words (``ReferenceCaption.NAMED``: ``--reference-caption named``);
- plain: the sample image, and the hand alone in words (``ReferenceCaption.HAND``: ``--reference-caption hand``);
- sample: the sample image alone, naming neither the hand nor the look, its paper "the paper of the sample image"
  (``ReferenceCaption.SAMPLE``: ``--reference-caption sample``), so both must be read from it: where each archive has
  one look, its name gives the hand and the look away and the sample need not be read;
- sample and look: the sample image, and the look in words but neither archive nor years (``ReferenceCaption.
  SAMPLE_LOOK``), so the hand must be read from it;
- generic: no reference, nothing named, neither archive, years nor look (``ReferenceCaption.GENERIC``): the page's
  content alone, the form drawn without the sample in the content mix, which names no hand in any form.

A set built before a form was added gets it for the rows of its manifests from ``diffu_page.data.page_pair_forms``, in
new manifests beside the old, its images as they are. ``diffu_page.data.page_pair_samples`` gives a manifest's rows
samples at their targets' writing scale (legible) in place of the shrunk ``refs/``, likewise in a new manifest.

A page's prompt is counted in the text encoder's tokens by MOSS's own tokenizer (``--model``'s); a page whose longest
form passes the encoder's 4,096 tokens is skipped and counted, never cut. A page's sample is another single page of its
volume (same collection and volume folder) sharing none of its lines' texts (compared by 8 bytes of each text's
digest), picked from one seeded stream; a page without one is skipped and counted. With ``--sample-pick nearest`` it is
the nearest such page instead: the other page of its own opening, else the scans closest before and after it in its
volume's file order (the same scribe far more often than a page drawn from the whole volume). Whole volumes are held out
(``--val-volumes``, a share of them, at least one): their pairs go to ``val.jsonl``, never to ``train.jsonl``, and
``--eval-pages`` of them spread over it make ``val_eval/``, a ``diffu_page.cli.evaluate`` dir (each page, its sample in
``refs/`` and the sample's line boxes; drawn with MOSS-Image, its caption compiles to the look form's prompt).

With ``--windows-per-page N`` each page also gives up to N windows: consecutive whole lines of one of its columns,
planned on the single page exactly as ``diffu-generate`` plans its crops (``plan_page``: a 64 px line pitch at a NeMo
bucket's size, 8-15 lines on real pages), so never across a spread, and picked among the page's windows by a stream
seeded with the page's id. A window is cut from the scan at about ``--megapixels`` as a page is (the same pixels,
larger letters) and prompted as its page is, in its page's style: the same four forms, each its own exact Layout
prompt. Each window is a pair row as a page is, with its page's sample, marked as a window
(``PagePairRow.window``: its page and its rectangle there); a held-out volume's windows go to ``val.jsonl``, and a
window whose longest form passes the encoder's tokens is skipped and counted. A page too long to be a pair still gives
its windows a sample, drawn after every page's from the same stream. The window rows follow the page rows, which are
written as a build without windows (the default) writes them.

A build holds none of its scans' pages as planned: as each scan is read it keeps its pages' entries alone
(:class:`PageEntry`: a page's id, volume and windows, whether a form of its prompt is too long, its lines' texts by
their digests), which its pages are grouped, paired and split by, and each manifest row is written from its scan's
record, read back as the row is written. Its memory grows with the scans by those entries and the pairs made of them,
about 1.6 KB a page (28 MB for 10,000 scans of the GT, where holding the pages as planned took 7.8 GB), and its
processes are given a few scans each ahead of the results taken.

Written under ``--out``:

- ``pages/<id>.jpg``, each page at its whole-page size, ``refs/<id>.jpg``, each at the reference size
  (``as_reference``, about 0.5 MP), and ``windows/<id>.jpg`` (``<page id>_w<k>``, window k of its page in page order),
  cut from the scan;
- ``scanned/<digest>.json``, a scan's pages and windows as planned (:class:`ScanResult`), written whole after its
  images, so a stopped build resumes where it stopped and its rows are written from them (``scan.json`` holds what they
  are cut and prompted by: a build resumed with other settings, or one recorded before windows, is refused);
- ``train.jsonl`` and ``val.jsonl``, the page-pair manifests (``diffu_moss.train.pairs.PagePairRow``, paths relative
  to ``--out``), ``val_eval/`` and, last, ``build.json`` (:class:`BuildRecord`: how the set was built and what became of
  every scan, page and window).

    python -m diffu_page.data.page_pairs --gt-root ~/Desktop/data/GT-Lejonet_HTR_202602 --model runs/moss/base \\
        --out data_flux3/page_pairs --workers 8 --windows-per-page 2
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import random
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from enum import StrEnum
from functools import cache, lru_cache
from pathlib import Path
from typing import Annotated, NamedTuple

import typer
from PIL import Image
from pydantic import BaseModel, ConfigDict, ValidationError
from transformers import PreTrainedTokenizerBase

from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.modular_pipelines.moss.encoders import MAX_PROMPT_TOKENS, MossTextEncoderStep, prompt_tokens
from diffu_moss.sample_caption import sample_layout_prompt
from diffu_moss.train.pairs import PagePairRow, PageWindow, Variant
from diffu_page.captions.look import scan_look
from diffu_page.captions.prompt import LineRecord, PageStyle
from diffu_page.captions.provenance import Provenance, provenance_of
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.data.build_crops import TILE_QUALITY
from diffu_page.data.evalset import ReferencedCrop, write_jsonl
from diffu_page.data.style_pairs import link
from diffu_page.draw.plan import CropPlan, Reference, plan_page, plan_whole_page, whole_page_size
from diffu_page.local_run import describe
from diffu_page.output import JPEG_QUALITY, console, load_rgb, validation_problems
from diffu_page.page.crops.geometry import regions_of
from diffu_page.page.crops.grid import as_reference
from diffu_page.page.layout import Layout, PageBox, ScanPage, is_unsplit_spread, load_xml_layout, scan_pages
from diffu_page.page.layout.page_xml import find_scan
from diffu_page.page.layout.pages import digest, page_id

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

SCANNED = "scanned"
PAGES = "pages"
REFS = "refs"
WINDOWS = "windows"
VAL_EVAL = "val_eval"
SCAN_RECORD = "scan.json"
BUILD_RECORD = "build.json"
IN_FLIGHT = 8  # tasks given each process ahead of the results taken
RECORDS_KEPT = 4  # scans' records held read while a build writes its rows: a scan's pages and windows follow each other
FORM_REFERENCES: dict[Variant, Reference | None] = {  # the sample is only named in these forms' captions, never read
    Variant.NO_REFERENCE: None,
    Variant.LOOK: Reference(image=Path("sample"), caption=ReferenceCaption.NAMED),
    Variant.PLAIN: Reference(image=Path("sample"), caption=ReferenceCaption.HAND),
    Variant.SAMPLE: Reference(image=Path("sample"), caption=ReferenceCaption.SAMPLE),
    Variant.SAMPLE_LOOK: Reference(image=Path("sample"), caption=ReferenceCaption.SAMPLE_LOOK),
    Variant.GENERIC: Reference(image=Path("sample"), caption=ReferenceCaption.GENERIC),  # cites none, names nothing
}


class ScanSettings(BaseModel):
    """What a build's pages and windows are cut and prompted by; a resumed build must be given the same."""

    model_config = ConfigDict(frozen=True)

    gt_root: str
    model: str  # the MOSS model directory whose tokenizer counts the prompts
    pixels: int  # a page's or a window's size, about
    windows_per_page: int  # at most; required, so a build recorded before windows (and spreads skipped) is not resumed


class Planned(BaseModel):
    """A page or a window as planned and prompted: its size, its lines, and its Layout prompt in every form with its
    length in the text encoder's tokens."""

    size: tuple[int, int]  # (width, height) px
    lines: list[LineRecord]
    prompts: dict[Variant, str]
    tokens: dict[Variant, int]

    def texts(self) -> frozenset[str]:
        """The texts of its lines."""
        return frozenset(line.text for line in self.lines)

    @property
    def too_long(self) -> bool:
        """Whether a form of its prompt passes what the text encoder reads."""
        return max(self.tokens.values()) > MAX_PROMPT_TOKENS


class WindowRecord(Planned):
    """A window of a page, planned as diffu-generate plans a page's crops: its id (its image is ``windows/<id>.jpg``)
    and its rectangle on its page, px of the page's own coordinates."""

    id: str
    rect: PageBox


class PageRecord(Planned):
    """One single page of a scan as planned whole: its id (its images are ``pages/<id>.jpg`` and ``refs/<id>.jpg``),
    its volume, the look form's caption as its plan gives it, and the windows cut of it."""

    id: str
    volume: str  # "<collection>/<volume>"
    caption: str  # the look form's caption, citing <ref_image_0>: what an eval row of the page stores
    windows: list[WindowRecord] = []

    def entry(self) -> PageEntry:
        """What a build groups, pairs and splits the page by: all it holds of it."""
        return PageEntry(
            id=self.id,
            volume=self.volume,
            too_long=self.too_long,
            digests=text_digests(self.texts()),
            windows=tuple(WindowEntry(id=window.id, too_long=window.too_long) for window in self.windows),
        )


class ScanProblem(StrEnum):
    """Why a scan gave no page."""

    NO_SCAN = "no scan"  # its PAGE file names no image beside it
    NO_TEXT = "no text"  # no page of it holds a line
    SPREAD = "spread"  # wider than tall, and its lines leave no gutter to split its pages at
    FAILED = "failed"  # reading or cutting it failed: its error is kept


class ScanResult(BaseModel):
    """What a PAGE file gave: its single pages, or why it gave none."""

    xml: str  # relative to the GT root
    pages: list[PageRecord] = []
    problem: ScanProblem | None = None
    error: str = ""

    def entry(self) -> ScanEntry:
        """What a build holds of the scan: its pages' entries, or why it gave none."""
        return ScanEntry(
            xml=self.xml, pages=tuple(page.entry() for page in self.pages), problem=self.problem, error=self.error
        )


def text_digests(texts: Iterable[str]) -> bytes:
    """The distinct ``texts``, each as 8 bytes of its BLAKE2b digest, sorted: what a build compares two pages' lines by
    (:class:`PageEntry`), in a tenth of their memory. Two texts share 8 bytes once in 2^64."""
    return b"".join(sorted({hashlib.blake2b(text.encode(), digest_size=8).digest() for text in texts}))


class WindowEntry(NamedTuple):
    """What a build pairs a window by (its scan's record holds its :class:`WindowRecord`): its id, and whether a form of
    its prompt passes what the text encoder reads."""

    id: str
    too_long: bool


class PageEntry(NamedTuple):
    """What a build groups, pairs and splits a page by, all it holds of it (its scan's record holds its
    :class:`PageRecord`, which its rows are written from): its id, its volume, whether a form of its prompt passes what
    the text encoder reads, its lines' texts by their digests (:func:`text_digests`) and its windows. A tuple, not a
    model: one is held for every page of a build, and a model takes five times the memory."""

    id: str
    volume: str
    too_long: bool
    digests: bytes
    windows: tuple[WindowEntry, ...]

    def texts(self) -> memoryview[int]:
        """Its lines' distinct texts, each as the integer of its digest: two pages share a line when they share one."""
        return memoryview(self.digests).cast("Q")  # 8 bytes a text


class ScanEntry(NamedTuple):
    """What a build holds of a scan (its record holds its :class:`ScanResult`): its PAGE file, its pages' entries, and
    why it gave none."""

    xml: str
    pages: tuple[PageEntry, ...]
    problem: ScanProblem | None
    error: str


class ScanTask(BaseModel):
    """A PAGE file to read into ``out``, by ``settings``."""

    xml: Path
    out: Path
    settings: ScanSettings


class BuildCounts(BaseModel):
    """What became of a build's scans, pages and windows."""

    scans: int
    no_scan: int
    no_text: int
    spread: int  # wider than tall, its pages not split: an opening whose lines overlap, or a single wide sheet
    failed: int
    pages: int
    too_long: int  # a form's prompt passes the text encoder's tokens
    no_partner: int  # no other page of its volume shares none of its lines
    train_pairs: int
    val_pairs: int
    windows: int  # cut of the pages, up to the windows per page of each
    windows_too_long: int
    windows_no_partner: int  # their page has no sample
    train_windows: int
    val_windows: int
    eval_pages: int


class BuildRecord(BaseModel):
    """``build.json``: what a page-pair set was built from and how, and what became of its scans, pages and windows."""

    scan: ScanSettings
    collections: list[str]
    max_scans: int | None
    val_volumes: float
    eval_pages: int
    seed: int
    held_out_volumes: list[str]
    counts: BuildCounts
    failed: dict[str, str]  # the scans whose reading failed, by PAGE file, with their error


class SamplePick(StrEnum):
    """How a page's sample is picked among the pages of its volume sharing none of its lines: drawn from all of them
    (``random``), or the nearest (``nearest``): the other page of its opening, else the scans closest before and after
    it in the volume's file order, drawn between two as near."""

    RANDOM = "random"
    NEAREST = "nearest"


class BuildSettings(BaseModel):
    """Which scans a build reads (``collections``, every one when empty; up to ``max_scans`` of each, picked by
    ``seed``), how its pages and windows are cut and prompted (``scan``), and how its pairs are split and picked
    (``sample_pick``)."""

    model_config = ConfigDict(frozen=True)

    scan: ScanSettings
    collections: list[str] = []
    max_scans: int | None = None
    val_volumes: float = 0.03
    eval_pages: int = 24
    seed: int = 0
    sample_pick: SamplePick = SamplePick.RANDOM


@cache
def tokenizer_of(model: str) -> PreTrainedTokenizerBase:
    """The tokenizer of the MOSS model directory ``model``, loaded as the page pipeline loads it, once a process."""
    pipe = MossModularPipeline(blocks=MossTextEncoderStep(), pretrained_model_name_or_path=model)
    pipe.load_components(names=["tokenizer"])
    if pipe.tokenizer is None:
        raise RuntimeError(f"{model}: its tokenizer did not load")
    return pipe.tokenizer


def form_prompts(plans: Mapping[Variant, CropPlan], size: tuple[int, int]) -> dict[Variant, str]:
    """A page's or a window's plan in each form compiled as MOSS's Layout prompt on a canvas of ``size`` (width,
    height) px, every line's exact text in its box, the sample image cited in the forms drawn with it."""
    texts = {line.id: line.text for line in plans[Variant.LOOK].lines}
    return {
        variant: sample_layout_prompt(plan.caption, texts, size, sample=variant.takes_reference)
        for variant, plan in plans.items()
    }


def token_counts(prompts: Mapping[Variant, str], tokenizer: PreTrainedTokenizerBase) -> dict[Variant, int]:
    """Each form's prompt's length in the text encoder's tokens."""
    return {variant: prompt_tokens(tokenizer, prompt) for variant, prompt in prompts.items()}


def planned_page(page: ScanPage, style: PageStyle, *, pixels: int, tokenizer: PreTrainedTokenizerBase) -> PageRecord:
    """``page`` planned whole at about ``pixels`` in ``style`` and prompted in every form: its record (its id and
    volume left blank)."""
    plans = {
        variant: plan_whole_page(page.layout, style, pixels=pixels, reference=reference)
        for variant, reference in FORM_REFERENCES.items()
    }
    look = plans[Variant.LOOK]
    size = (look.width, look.height)
    prompts = form_prompts(plans, size)
    return PageRecord(
        id="",
        volume="",
        size=size,
        lines=look.lines,
        caption=look.caption,
        prompts=prompts,
        tokens=token_counts(prompts, tokenizer),
    )


def planned_windows(
    page: ScanPage, style: PageStyle, identity: str, *, settings: ScanSettings, tokenizer: PreTrainedTokenizerBase
) -> list[WindowRecord]:
    """Up to ``settings.windows_per_page`` of the windows of ``page`` (its id ``identity``), planned in ``style`` as
    diffu-generate plans a page's crops (``plan_page``) and picked among them by a stream seeded with its id, each sized
    at about ``settings.pixels`` on the latent grid and prompted in every form: their records, in page order."""
    if not settings.windows_per_page:
        return []
    plans = {
        variant: plan_page(page.layout, style, reference=reference) for variant, reference in FORM_REFERENCES.items()
    }
    count = len(plans[Variant.LOOK])
    picked = sorted(random.Random(identity).sample(range(count), min(settings.windows_per_page, count)))
    windows = []
    for k in picked:
        window = {variant: plans[variant][k] for variant in plans}
        x0, y0, x1, y1 = window[Variant.LOOK].rect
        size = whole_page_size(round(x1 - x0), round(y1 - y0), settings.pixels)
        prompts = form_prompts(window, size)
        windows.append(
            WindowRecord(
                id=f"{identity}_w{k}",
                rect=window[Variant.LOOK].rect,
                size=size,
                lines=window[Variant.LOOK].lines,
                prompts=prompts,
                tokens=token_counts(prompts, tokenizer),
            )
        )
    return windows


def load_scan(path: Path, layout: Layout) -> Image.Image:
    """The scan at ``path`` of the PAGE file read as ``layout``, whole and as RGB.

    Raises:
        ValueError: If it is not the size its PAGE file gives.
    """
    Image.MAX_IMAGE_PIXELS = None  # archival spreads exceed PIL's decompression-bomb default
    scan = load_rgb(path)
    if scan.size != (layout.width, layout.height):
        raise ValueError(f"{path.name} is {scan.size} px, its PAGE file {(layout.width, layout.height)}")
    return scan


def whole_page(page: ScanPage, cut: Image.Image, pixels: int) -> Image.Image:
    """``cut``, ``page`` cut from its scan, at the page's whole-page size (about ``pixels``, each side on the latent
    grid): the page's target, which its look is measured on."""
    return cut.resize(whole_page_size(page.layout.width, page.layout.height, pixels), Image.Resampling.LANCZOS)


def page_record(
    page: ScanPage, image: Image.Image, identity: str, *, provenance: Provenance, settings: ScanSettings
) -> PageRecord:
    """``page`` (its id ``identity``) captioned in its archive's name (``provenance``) and the look measured on
    ``image``, the page at its whole-page size (:func:`whole_page`), planned whole (:func:`planned_page`) and its
    windows planned (:func:`planned_windows`), every form prompted: its record (its volume left blank)."""
    tokenizer = tokenizer_of(settings.model)
    style = PageStyle(provenance=provenance, look=scan_look(image))
    record = planned_page(page, style, pixels=settings.pixels, tokenizer=tokenizer)
    windows = planned_windows(page, style, identity, settings=settings, tokenizer=tokenizer)
    return record.model_copy(update={"id": identity, "windows": windows})


def written_page(
    page: ScanPage, cut: Image.Image, task: ScanTask, *, identity: str, volume: str, provenance: Provenance
) -> PageRecord:
    """``page`` (its id ``identity``, of ``volume``), ``cut`` from its scan, planned and prompted at its whole-page
    size (:func:`page_record`): its images and its windows' written under ``task.out``, and its record."""
    image = whole_page(page, cut, task.settings.pixels)
    record = page_record(page, image, identity, provenance=provenance, settings=task.settings)
    image.save(task.out / PAGES / f"{identity}.jpg", quality=TILE_QUALITY)
    as_reference(cut).save(task.out / REFS / f"{identity}.jpg", quality=JPEG_QUALITY)
    for window in record.windows:
        window_image = cut.crop(window.rect).resize(window.size, Image.Resampling.LANCZOS)
        window_image.save(task.out / WINDOWS / f"{window.id}.jpg", quality=TILE_QUALITY)
    return record.model_copy(update={"volume": volume})


def scanned(task: ScanTask) -> ScanResult:
    """A PAGE file's single pages cut and planned, with their windows (:func:`written_page`), their images written
    under ``task.out``.

    Raises:
        ValueError: If the PAGE file cannot be read, or its scan is not the size it gives.
    """
    root = Path(task.settings.gt_root)
    xml = task.xml.relative_to(root)
    scan_path = find_scan(task.xml)
    if scan_path is None:
        return ScanResult(xml=xml.as_posix(), problem=ScanProblem.NO_SCAN)
    layout = load_xml_layout(task.xml)
    pages = [page for page in scan_pages(layout) if regions_of(page.layout)]  # a page holding a line of text
    if not pages:
        return ScanResult(xml=xml.as_posix(), problem=ScanProblem.NO_TEXT)
    if is_unsplit_spread(layout):
        return ScanResult(xml=xml.as_posix(), problem=ScanProblem.SPREAD)
    scan = load_scan(scan_path, layout)
    collection, volume = xml.parts[:2]
    provenance = provenance_of(collection, task.xml)
    written = [
        written_page(
            page,
            scan.crop(page.rect),
            task,
            identity=page_id(xml, page),
            volume=f"{collection}/{volume}",
            provenance=provenance,
        )
        for page in pages
    ]
    return ScanResult(xml=xml.as_posix(), pages=written)


def scan_once(task: ScanTask) -> ScanEntry:
    """A PAGE file's entry (:meth:`ScanResult.entry`): of the result a stopped build recorded, else of :func:`scanned`'s,
    recorded first (written whole before it is named): the build's rows are written from the record. A scan that fails
    is recorded with its error, so one unreadable page does not end the build."""
    xml = task.xml.relative_to(Path(task.settings.gt_root)).as_posix()
    record = task.out / SCANNED / f"{digest(xml)}.json"
    if record.exists():
        return ScanResult.model_validate_json(record.read_text(encoding="utf-8")).entry()
    try:
        result = scanned(task)
    except Exception as error:  # one unreadable page must not end the build; it is recorded and counted
        result = ScanResult(xml=xml, problem=ScanProblem.FAILED, error=describe(error))
    partial = record.with_suffix(".partial")
    partial.write_text(result.model_dump_json(), encoding="utf-8")
    partial.rename(record)
    return result.entry()


def page_files(gt_root: Path, collections: list[str], *, max_scans: int | None, seed: int) -> list[Path]:
    """The PAGE files to read: every one of each collection (all when ``collections`` is empty), macOS ``._*`` files
    passed over, or ``max_scans`` of each picked by ``seed``; collection by collection, sorted."""
    names = collections or sorted(path.name for path in gt_root.iterdir() if path.is_dir())
    files: list[Path] = []
    for name in names:
        found = sorted(path for path in (gt_root / name).rglob("page/*.xml") if not path.name.startswith("._"))
        if max_scans is not None and len(found) > max_scans:
            found = sorted(random.Random(f"{seed}/{name}").sample(found, max_scans))
        files += found
    return files


def in_processes[T, R](work: Callable[[T], R], tasks: Iterable[T], workers: int) -> Iterator[R]:
    """``work`` done on every task, its results in order, on ``workers`` processes (in this one for one). The tasks are
    drawn as the results are taken, :data:`IN_FLIGHT` a process ahead, so neither every task nor every result waiting
    to be taken is held at once."""
    if workers == 1:
        yield from map(work, tasks)
        return
    # forkserver: a process holding threads (torch, the tokenizer) is not forked
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        waiting: deque[Future[R]] = deque()
        try:
            for task in tasks:
                waiting.append(pool.submit(work, task))
                if len(waiting) == IN_FLIGHT * workers:
                    yield waiting.popleft().result()
            while waiting:
                yield waiting.popleft().result()
        finally:  # a stopped run waits for the tasks running, not for those waiting
            for future in waiting:
                future.cancel()


def held_out_volumes(volumes: Iterable[str], share: float, seed: int) -> set[str]:
    """The volumes held out: ``share`` of them (rounded, at least one of two or more when ``share`` is not 0), ranked
    by a digest of ``seed`` and their name, so a volume's split does not depend on which others are read."""
    ranked = sorted(set(volumes), key=lambda volume: digest(f"{seed}/{volume}"))
    count = round(len(ranked) * share)
    if share and len(ranked) > 1:
        count = max(1, count)
    return set(ranked[:count])


def pick_partner(
    page: PageEntry, volume: list[PageEntry], rng: random.Random, places: Mapping[str, int] | None = None
) -> PageEntry | None:
    """Another page of ``page``'s ``volume`` (its pages, in a fixed order) sharing none of its lines' texts, drawn from
    ``rng``; None when there is none. A sample holding the page's own lines teaches copying them. With ``places`` (each
    page's place in ``volume``, which is then in its file order: :func:`file_places`) the nearest such page: the other
    page of its opening, else the closest before or after, ``rng`` drawing between two as near."""
    texts = set(page.texts())
    if places is None:
        others = [other for other in volume if other.id != page.id]
        rng.shuffle(others)
        return next((other for other in others if texts.isdisjoint(other.texts())), None)
    at = places[page.id]
    for distance in range(1, len(volume)):
        near = [volume[place] for place in (at - distance, at + distance) if 0 <= place < len(volume)]
        rng.shuffle(near)
        near.sort(key=lambda other: scan_digest(other.id) != scan_digest(page.id))  # its opening's other page first
        found = next((other for other in near if texts.isdisjoint(other.texts())), None)
        if found is not None:
            return found
    return None


def file_places(volumes: dict[str, list[PageEntry]], files: Mapping[str, str]) -> dict[str, int]:
    """Each volume's pages put in their file order in place (by the PAGE file each page is of, ``files`` by page id,
    and its side: an opening's left page before its right), and each page's place in its volume."""
    places: dict[str, int] = {}
    for pages in volumes.values():
        pages.sort(key=lambda page: (files[page.id], page.id))
        places.update((page.id, place) for place, page in enumerate(pages))
    return places


def window_partners(
    pages: list[PageEntry],
    volumes: Mapping[str, list[PageEntry]],
    rng: random.Random,
    places: Mapping[str, int] | None = None,
) -> dict[str, PageEntry]:
    """A sample for each page too long to be a pair that has a window short enough to be one, by page id, drawn from
    ``rng`` (:func:`pick_partner`, the nearest with ``places``) after every page's own."""
    partners: dict[str, PageEntry] = {}
    for page in pages:
        if page.too_long and any(not window.too_long for window in page.windows):
            partner = pick_partner(page, volumes[page.volume], rng, places)
            if partner is not None:
                partners[page.id] = partner
    return partners


def window_pairs(
    pages: list[PageEntry], partners: Mapping[str, PageEntry]
) -> list[tuple[WindowEntry, PageEntry, PageEntry]]:
    """Each window short enough for the text encoder whose page has a sample in ``partners``, with its page and that
    sample, in page order."""
    return [
        (window, page, partners[page.id])
        for page in pages
        if page.id in partners
        for window in page.windows
        if not window.too_long
    ]


def scan_digest(identity: str) -> str:
    """The digest of the PAGE file of the scan a page or a window (its id ``identity``) is of, which its id begins with
    (:func:`page_id`): the scan's record is ``scanned/<digest>.json``."""
    return identity.split("_", 1)[0]


def record_reader(out: Path) -> Callable[[str], PageRecord]:
    """The pages of the build in ``out`` by their ids, each read from its scan's record, the last
    :data:`RECORDS_KEPT` scans' records kept: a build's rows are written in page order, so the rows of a scan's pages,
    and of their windows, follow each other."""

    @lru_cache(maxsize=RECORDS_KEPT)
    def scan(name: str) -> ScanResult:
        return ScanResult.model_validate_json((out / SCANNED / f"{name}.json").read_text(encoding="utf-8"))

    def page(identity: str) -> PageRecord:
        return next(page for page in scan(scan_digest(identity)).pages if page.id == identity)

    return page


def pair_row(target: PageRecord, reference: str) -> PagePairRow:
    """A page and its sample (its page's id ``reference``) as a manifest row, its images under ``pages/`` and
    ``refs/``."""
    return PagePairRow(
        id=target.id,
        target=f"{PAGES}/{target.id}.jpg",
        reference=f"{REFS}/{reference}.jpg",
        size=target.size,
        volume=target.volume,
        prompts=target.prompts,
        tokens=target.tokens,
    )


def window_row(window: WindowRecord, page: PageRecord, reference: str) -> PagePairRow:
    """A window of ``page`` and the page's sample (its page's id ``reference``) as a manifest row, its image under
    ``windows/``, marked as a window of the page."""
    return PagePairRow(
        id=window.id,
        target=f"{WINDOWS}/{window.id}.jpg",
        reference=f"{REFS}/{reference}.jpg",
        size=window.size,
        volume=page.volume,
        prompts=window.prompts,
        tokens=window.tokens,
        window=PageWindow(page=page.id, rect=window.rect),
    )


def manifest_rows(
    pairs: Iterable[tuple[PageEntry, PageEntry]],
    windows: Iterable[tuple[WindowEntry, PageEntry, PageEntry]],
    page_of: Callable[[str], PageRecord],
) -> Iterator[PagePairRow]:
    """A manifest's rows: each page of ``pairs`` with its sample, then each window of ``windows`` with its page's
    sample, each page read (``page_of``) as its row or its window's is made."""
    for target, reference in pairs:
        yield pair_row(page_of(target.id), reference.id)
    for window, page, reference in windows:
        record = page_of(page.id)
        yield window_row(next(item for item in record.windows if item.id == window.id), record, reference.id)


def write_rows(path: Path, rows: Iterable[BaseModel]) -> None:
    """``rows`` as the JSON-lines file at ``path`` (as ``evalset.write_jsonl`` writes them), each written as it comes:
    none held but the one."""
    with path.open("w", encoding="utf-8") as lines:
        lines.writelines(f"{row.model_dump_json()}\n" for row in rows)


def eval_row(target: PageRecord, reference: PageRecord) -> ReferencedCrop:
    """A held-out page as ``diffu_page.cli.evaluate`` draws it: its look form's caption and lines, its sample beside
    it with the sample's line boxes."""
    return ReferencedCrop(
        file_name=f"{target.id}.jpg",
        caption=target.caption,
        width=target.size[0],
        height=target.size[1],
        lines=target.lines,
        whole_page=True,
        reference=f"{REFS}/{reference.id}.jpg",
        reference_lines=[line.bbox for line in reference.lines],
    )


def write_eval_set(
    out: Path, pairs: list[tuple[PageEntry, PageEntry]], count: int, *, page_of: Callable[[str], PageRecord]
) -> int:
    """``count`` of the held-out ``pairs``, spread over them, as ``out/val_eval``, their images hard-linked there and
    their pages read (``page_of``) for their rows.

    Returns:
        How many pages it holds.
    """
    spread = pairs[:: max(1, len(pairs) // count)][:count] if count else []
    directory = out / VAL_EVAL
    (directory / REFS).mkdir(parents=True, exist_ok=True)
    for target, reference in spread:
        link(out / PAGES / f"{target.id}.jpg", directory / f"{target.id}.jpg")
        link(out / REFS / f"{reference.id}.jpg", directory / REFS / f"{reference.id}.jpg")
    rows = [eval_row(page_of(target.id), page_of(reference.id)) for target, reference in spread]
    write_jsonl(directory / "metadata.jsonl", rows)
    return len(spread)


def recorded_settings(scan_record: Path) -> ScanSettings:
    """The settings the stopped build whose ``scan.json`` is ``scan_record`` was cut and prompted by.

    Raises:
        ValueError: If it records other settings than this builder's: a build from before windows were cut and spreads
            skipped, which may hold spreads.
    """
    try:
        return ScanSettings.model_validate_json(scan_record.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ValueError(
            f"{scan_record} is not this builder's ({validation_problems(error, SCAN_RECORD)}): a build from before "
            "windows were cut and spreads skipped is not resumed; give a new dir"
        ) from error


def build(out: Path, settings: BuildSettings, *, workers: int) -> BuildRecord:
    """The page pairs of ``settings``' scans built under ``out`` (the module's docstring), resuming a stopped build.

    Returns:
        The build's record, also written to ``out/build.json``.

    Raises:
        ValueError: If ``out`` holds a build cut or prompted by other settings.
    """
    scan_record = out / SCAN_RECORD
    if scan_record.exists():
        earlier = recorded_settings(scan_record)
        if earlier != settings.scan:
            raise ValueError(f"{out} holds a build of {earlier}; resume it with those settings or give a new dir")
    for directory in (SCANNED, PAGES, REFS, *([WINDOWS] if settings.scan.windows_per_page else [])):
        (out / directory).mkdir(parents=True, exist_ok=True)
    scan_record.write_text(settings.scan.model_dump_json(indent=1), encoding="utf-8")

    root = Path(settings.scan.gt_root)
    # as text, in their order: a sorted Path keeps its parts, a kilobyte a scan more
    files = [
        path.as_posix()
        for path in page_files(root, settings.collections, max_scans=settings.max_scans, seed=settings.seed)
    ]
    tasks = (ScanTask(xml=Path(xml), out=out, settings=settings.scan) for xml in files)
    pages: list[PageEntry] = []
    page_xml: dict[str, str] = {}  # each page's PAGE file: a volume's file order, which nearest samples are picked by
    outcomes: Counter[ScanProblem | None] = Counter()  # each scan's problem, None for a scan that gave pages
    failed: dict[str, str] = {}
    for k, scan in enumerate(in_processes(scan_once, tasks, workers), start=1):
        pages += scan.pages
        page_xml.update((page.id, scan.xml) for page in scan.pages)
        outcomes[scan.problem] += 1
        if scan.problem is ScanProblem.FAILED:
            failed[scan.xml] = scan.error
        if k % 500 == 0:
            console.print(f"{k} of {len(files)} scans read")

    pages.sort(key=lambda page: page.id)
    volumes: dict[str, list[PageEntry]] = defaultdict(list)
    for page in pages:
        volumes[page.volume].append(page)
    held_out = held_out_volumes(volumes, settings.val_volumes, settings.seed)
    places = file_places(volumes, page_xml) if settings.sample_pick is SamplePick.NEAREST else None
    rng = random.Random(settings.seed)
    train: list[tuple[PageEntry, PageEntry]] = []
    val: list[tuple[PageEntry, PageEntry]] = []
    partners: dict[str, PageEntry] = {}
    too_long = no_partner = 0
    for page in pages:
        if page.too_long:
            too_long += 1
            continue
        partner = pick_partner(page, volumes[page.volume], rng, places)
        if partner is None:
            no_partner += 1
            continue
        partners[page.id] = partner
        (val if page.volume in held_out else train).append((page, partner))
    windows = window_pairs(pages, partners | window_partners(pages, volumes, rng, places))
    train_windows = [pair for pair in windows if pair[1].volume not in held_out]
    val_windows = [pair for pair in windows if pair[1].volume in held_out]
    page_of = record_reader(out)
    write_rows(out / "train.jsonl", manifest_rows(train, train_windows, page_of))
    write_rows(out / "val.jsonl", manifest_rows(val, val_windows, page_of))
    eval_pages = write_eval_set(out, val, settings.eval_pages, page_of=page_of)

    cut = [window for page in pages for window in page.windows]
    windows_too_long = sum(window.too_long for window in cut)
    record = BuildRecord(
        scan=settings.scan,
        collections=settings.collections,
        max_scans=settings.max_scans,
        val_volumes=settings.val_volumes,
        eval_pages=settings.eval_pages,
        seed=settings.seed,
        held_out_volumes=sorted(held_out),
        counts=BuildCounts(
            scans=outcomes.total(),
            no_scan=outcomes[ScanProblem.NO_SCAN],
            no_text=outcomes[ScanProblem.NO_TEXT],
            spread=outcomes[ScanProblem.SPREAD],
            failed=outcomes[ScanProblem.FAILED],
            pages=len(pages),
            too_long=too_long,
            no_partner=no_partner,
            train_pairs=len(train),
            val_pairs=len(val),
            windows=len(cut),
            windows_too_long=windows_too_long,
            windows_no_partner=len(cut) - windows_too_long - len(windows),
            train_windows=len(train_windows),
            val_windows=len(val_windows),
            eval_pages=eval_pages,
        ),
        failed=failed,
    )
    (out / BUILD_RECORD).write_text(record.model_dump_json(indent=1), encoding="utf-8")
    return record


@app.command()
def main(
    gt_root: Annotated[Path, typer.Option(help="the GT folder, one subfolder per collection")],
    model: Annotated[Path, typer.Option(help="a MOSS model directory (diffu-moss-init): its tokenizer counts prompts")],
    out: Annotated[Path, typer.Option(help="the set's dir: a new one, or a stopped build's to resume")],
    collections: Annotated[
        list[str] | None, typer.Option("--collection", help="a collection to read (repeat it; default: all)")
    ] = None,
    max_scans: Annotated[
        int | None, typer.Option(min=1, help="read this many PAGE files of each collection, picked by --seed")
    ] = None,
    megapixels: Annotated[
        float, typer.Option(min=0.25, max=4.0, help="a page's and a window's size, as drawing plans a page")
    ] = 1.0,
    windows_per_page: Annotated[
        int,
        typer.Option(
            min=0,
            help="also cut up to this many windows of each page (whole lines of a column, as diffu-generate plans its "
            "crops), each a pair with its page's sample",
        ),
    ] = 0,
    val_volumes: Annotated[float, typer.Option(min=0.0, max=1.0, help="the share of volumes held out")] = 0.03,
    eval_pages: Annotated[int, typer.Option(min=0, help="held-out pages in val_eval/ for diffu-evaluate")] = 24,
    workers: Annotated[int, typer.Option(min=1, help="processes reading the scans: fewer beside a training")] = max(
        1, (os.cpu_count() or 2) // 2
    ),
    seed: Annotated[int, typer.Option(help="picks the scans read, the held-out volumes and each page's sample")] = 0,
    sample_pick: Annotated[
        SamplePick,
        typer.Option(
            help="each page's sample: random (any page of its volume sharing none of its lines) or nearest (the other "
            "page of its opening, else the closest scan before or after it: the same scribe more often)"
        ),
    ] = SamplePick.RANDOM,
) -> None:
    """Build page pairs for MOSS-Image's reference training: whole single pages, each with a sample page, and windows
    of them if asked."""
    root = gt_root.expanduser()
    settings = BuildSettings(
        scan=ScanSettings(
            gt_root=str(root), model=str(model), pixels=round(megapixels * 1_000_000), windows_per_page=windows_per_page
        ),
        collections=collections or [],
        max_scans=max_scans,
        val_volumes=val_volumes,
        eval_pages=eval_pages,
        seed=seed,
        sample_pick=sample_pick,
    )
    try:
        record = build(out, settings, workers=workers)
    except ValueError as error:  # a dir holding another build: the arguments' fault
        raise typer.BadParameter(str(error), param_hint="--out") from error
    console.print(f"{record.counts.model_dump_json()} -> {out}")
    console.print(f"held out: {', '.join(record.held_out_volumes)}")


if __name__ == "__main__":
    app()
