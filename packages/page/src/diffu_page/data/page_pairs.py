"""Page pairs for MOSS-Image's reference training: whole single pages of the handwritten ground truth, each with a
sample page of its own volume.

Every PAGE file of the GT (``<root>/<collection>/<volume>/<volume>/page/<file>.xml``, macOS ``._*`` files passed over)
is read with its scan (``find_scan``) and split into its single pages (``scan_pages``: an opening's two, else the scan
whole). A scan wider than tall that its lines cannot split (``is_unsplit_spread``: an opening whose pages' lines overlap
side to side, or a single wide sheet, which the lines cannot tell apart) gives no page: it is skipped and counted, as
no spread is ever trained on. Each page is cut from its scan at about ``--megapixels`` on the latent grid, exactly as ``diffu-generate
--whole-page`` plans a page (``plan_whole_page``), and prompted as drawing prompts it: the caption of its whole-page
plan compiled as MOSS's Layout prompt (``diffu_moss.sample_caption.sample_layout_prompt``, as ``MossLayoutPromptStep``
compiles it), every line's exact text in its box. The page is captioned in its own style: the archive and the volume's
years (``provenance_of``: whose hand it is, by name, as every caption names an archive) and the look measured on the
page (``scan_look``: greyscale or colour, the paper's and the ink's colour). A sample page carries paper and ink, not the
hand, so every form names the hand; each row carries the page's prompt in all three forms the training mixes
(``diffu_moss.train.pairs.Variant``; ``diffu-moss-cache-pairs`` fixes each pair's by its seed):

- no reference: the page's own caption (``plan_whole_page`` without a reference), hand and look in words;
- look: the sample image, and hand and look in words (``ReferenceCaption.NAMED``: ``--reference-caption named``);
- plain: the sample image, and the hand alone in words (``ReferenceCaption.HAND``: ``--reference-caption hand``).

A page's prompt is counted in the text encoder's tokens by MOSS's own tokenizer (``--model``'s); a page whose longest
form passes the encoder's 4,096 tokens is skipped and counted, never cut. A page's sample is another single page of its
volume (same collection and volume folder) sharing none of its lines' texts, picked from one seeded stream; a page
without one is skipped and counted. Whole volumes are held out (``--val-volumes``, a share of them, at least one): their
pairs go to ``val.jsonl``, never to ``train.jsonl``, and ``--eval-pages`` of them spread over it make ``val_eval/``, a
``diffu_page.cli.evaluate`` dir (each page, its sample in ``refs/`` and the sample's line boxes; drawn with MOSS-Image,
its caption compiles to the look form's prompt).

Written under ``--out``:

- ``pages/<id>.jpg``, each page at its whole-page size, and ``refs/<id>.jpg``, each at the reference size
  (``as_reference``, about 0.5 MP), cut from the scan;
- ``scanned/<digest>.json``, a scan's pages as planned (:class:`ScanResult`), written whole after its images, so a
  stopped build resumes where it stopped (``scan.json`` holds what pages are cut and prompted by: a build resumed with
  other settings is refused);
- ``train.jsonl`` and ``val.jsonl``, the page-pair manifests (``diffu_moss.train.pairs.PagePairRow``, paths relative
  to ``--out``), ``val_eval/`` and, last, ``build.json`` (:class:`BuildRecord`: how the set was built and what became of
  every scan and page).

    python -m diffu_page.data.page_pairs --gt-root ~/Desktop/data/GT-Lejonet_HTR_202602 --model runs/moss/base \\
        --out data_flux3/page_pairs --workers 8
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedTokenizerBase

from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.modular_pipelines.moss.encoders import MAX_PROMPT_TOKENS, MossTextEncoderStep, prompt_tokens
from diffu_moss.sample_caption import sample_layout_prompt
from diffu_moss.train.pairs import PagePairRow, Variant
from diffu_page.captions.look import scan_look
from diffu_page.captions.prompt import LineRecord, PageStyle
from diffu_page.captions.provenance import Provenance, provenance_of
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.crops.geometry import regions_of
from diffu_page.crops.grid import as_reference
from diffu_page.data.build_crops import TILE_QUALITY
from diffu_page.data.style_pairs import link
from diffu_page.evalset import ReferencedCrop, write_jsonl
from diffu_page.layout import ScanPage, is_unsplit_spread, load_xml_layout, scan_pages
from diffu_page.layout.page_xml import find_scan
from diffu_page.local_run import describe
from diffu_page.output import JPEG_QUALITY, console, load_rgb
from diffu_page.plan import Reference, plan_whole_page, whole_page_size

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

SCANNED = "scanned"
PAGES = "pages"
REFS = "refs"
VAL_EVAL = "val_eval"
SCAN_RECORD = "scan.json"
BUILD_RECORD = "build.json"
CAPTION_FORMS = {Variant.LOOK: ReferenceCaption.NAMED, Variant.PLAIN: ReferenceCaption.HAND}  # with the sample image


class ScanSettings(BaseModel):
    """What a build's pages are cut and prompted by; a resumed build must be given the same."""

    model_config = ConfigDict(frozen=True)

    gt_root: str
    model: str  # the MOSS model directory whose tokenizer counts the prompts
    pixels: int  # a page's size, about


class PageRecord(BaseModel):
    """One single page of a scan as planned whole: its id (its images are ``pages/<id>.jpg`` and ``refs/<id>.jpg``),
    its volume, its size, its lines and the look form's caption as its plan gives them, and its Layout prompt in every
    form with its length in the text encoder's tokens."""

    id: str
    volume: str  # "<collection>/<volume>"
    size: tuple[int, int]  # (width, height) px
    lines: list[LineRecord]
    caption: str  # the look form's caption, citing <ref_image_0>: what an eval row of the page stores
    prompts: dict[Variant, str]
    tokens: dict[Variant, int]

    def texts(self) -> frozenset[str]:
        """The texts of the page's lines."""
        return frozenset(line.text for line in self.lines)

    @property
    def too_long(self) -> bool:
        """Whether a form of its prompt passes what the text encoder reads."""
        return max(self.tokens.values()) > MAX_PROMPT_TOKENS


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


class ScanTask(BaseModel):
    """A PAGE file to read into ``out``, by ``settings``."""

    xml: Path
    out: Path
    settings: ScanSettings


class BuildCounts(BaseModel):
    """What became of a build's scans and pages."""

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
    eval_pages: int


class BuildRecord(BaseModel):
    """``build.json``: what a page-pair set was built from and how, and what became of its scans and pages."""

    scan: ScanSettings
    collections: list[str]
    max_scans: int | None
    val_volumes: float
    eval_pages: int
    seed: int
    held_out_volumes: list[str]
    counts: BuildCounts
    failed: dict[str, str]  # the scans whose reading failed, by PAGE file, with their error


class BuildSettings(BaseModel):
    """Which scans a build reads (``collections``, every one when empty; up to ``max_scans`` of each, picked by
    ``seed``), how its pages are cut and prompted (``scan``), and how its pairs are split and picked."""

    model_config = ConfigDict(frozen=True)

    scan: ScanSettings
    collections: list[str] = []
    max_scans: int | None = None
    val_volumes: float = 0.03
    eval_pages: int = 24
    seed: int = 0


@cache
def tokenizer_of(model: str) -> PreTrainedTokenizerBase:
    """The tokenizer of the MOSS model directory ``model``, loaded as the page pipeline loads it, once a process."""
    pipe = MossModularPipeline(blocks=MossTextEncoderStep(), pretrained_model_name_or_path=model)
    pipe.load_components(names=["tokenizer"])
    if pipe.tokenizer is None:
        raise RuntimeError(f"{model}: its tokenizer did not load")
    return pipe.tokenizer


def digest(text: str) -> str:
    """A stable 12-hex-digit name for ``text``."""
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def page_id(xml: Path, page: ScanPage) -> str:
    """A page's id: its PAGE file's digest, and its side of an opening."""
    return digest(xml.as_posix()) + (f"_{page.side}" if page.side else "")


def planned_page(
    page: ScanPage, cut: Image.Image, *, provenance: Provenance, pixels: int, tokenizer: PreTrainedTokenizerBase
) -> tuple[Image.Image, PageRecord]:
    """``page``, ``cut`` from its scan, planned whole at about ``pixels``, captioned in its archive's name and its
    measured look, and prompted in every form: its image at its whole-page size and its record (its id and volume left
    blank)."""
    size = whole_page_size(page.layout.width, page.layout.height, pixels)
    image = cut.resize(size, Image.Resampling.LANCZOS)
    style = PageStyle(provenance=provenance, look=scan_look(image))
    plans = {Variant.NO_REFERENCE: plan_whole_page(page.layout, style, pixels=pixels)}
    for variant, form in CAPTION_FORMS.items():  # the sample is not read for these forms, only named
        plans[variant] = plan_whole_page(
            page.layout, style, pixels=pixels, reference=Reference(image=Path("sample"), caption=form)
        )
    lines = plans[Variant.LOOK].lines
    texts = {line.id: line.text for line in lines}
    prompts = {
        variant: sample_layout_prompt(plan.caption, texts, size, sample=variant.takes_reference)
        for variant, plan in plans.items()
    }
    record = PageRecord(
        id="",
        volume="",
        size=size,
        lines=lines,
        caption=plans[Variant.LOOK].caption,
        prompts=prompts,
        tokens={variant: prompt_tokens(tokenizer, prompt) for variant, prompt in prompts.items()},
    )
    return image, record


def scanned(task: ScanTask) -> ScanResult:
    """A PAGE file's single pages cut and planned (:func:`planned_page`), their images written under ``task.out``.

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
    Image.MAX_IMAGE_PIXELS = None  # archival spreads exceed PIL's decompression-bomb default
    scan = load_rgb(scan_path)
    if scan.size != (layout.width, layout.height):
        raise ValueError(f"{scan_path.name} is {scan.size} px, its PAGE file {(layout.width, layout.height)}")
    collection, volume = xml.parts[:2]
    provenance = provenance_of(collection, task.xml)
    tokenizer = tokenizer_of(task.settings.model)
    records = []
    for page in pages:
        cut = scan.crop(page.rect)
        image, planned = planned_page(
            page, cut, provenance=provenance, pixels=task.settings.pixels, tokenizer=tokenizer
        )
        record = planned.model_copy(update={"id": page_id(xml, page), "volume": f"{collection}/{volume}"})
        image.save(task.out / PAGES / f"{record.id}.jpg", quality=TILE_QUALITY)
        as_reference(cut).save(task.out / REFS / f"{record.id}.jpg", quality=JPEG_QUALITY)
        records.append(record)
    return ScanResult(xml=xml.as_posix(), pages=records)


def scan_once(task: ScanTask) -> ScanResult:
    """A PAGE file's result: the one a stopped build recorded, else :func:`scanned`'s, recorded (written whole before
    it is named). A scan that fails is recorded with its error, so one unreadable page does not end the build."""
    xml = task.xml.relative_to(Path(task.settings.gt_root)).as_posix()
    record = task.out / SCANNED / f"{digest(xml)}.json"
    if record.exists():
        return ScanResult.model_validate_json(record.read_text(encoding="utf-8"))
    try:
        result = scanned(task)
    except Exception as error:  # one unreadable page must not end the build; it is recorded and counted
        result = ScanResult(xml=xml, problem=ScanProblem.FAILED, error=describe(error))
    partial = record.with_suffix(".partial")
    partial.write_text(result.model_dump_json(), encoding="utf-8")
    partial.rename(record)
    return result


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


def scan_all(tasks: list[ScanTask], workers: int) -> Iterator[ScanResult]:
    """Every task's result (:func:`scan_once`), in order, on ``workers`` processes (in this one for one)."""
    if workers == 1:
        yield from map(scan_once, tasks)
        return
    # forkserver: a process holding threads (torch, the tokenizer) is not forked
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        yield from pool.map(scan_once, tasks, chunksize=4)


def held_out_volumes(volumes: Iterable[str], share: float, seed: int) -> set[str]:
    """The volumes held out: ``share`` of them (rounded, at least one of two or more when ``share`` is not 0), ranked
    by a digest of ``seed`` and their name, so a volume's split does not depend on which others are read."""
    ranked = sorted(set(volumes), key=lambda volume: digest(f"{seed}/{volume}"))
    count = round(len(ranked) * share)
    if share and len(ranked) > 1:
        count = max(1, count)
    return set(ranked[:count])


def pick_partner(page: PageRecord, volume: list[PageRecord], rng: random.Random) -> PageRecord | None:
    """Another page of ``page``'s ``volume`` (its pages, in a fixed order) sharing none of its lines' texts, drawn from
    ``rng``; None when there is none. A sample holding the page's own lines teaches copying them."""
    texts = page.texts()
    others = [other for other in volume if other.id != page.id]
    rng.shuffle(others)
    return next((other for other in others if not texts & other.texts()), None)


def pair_row(target: PageRecord, reference: PageRecord) -> PagePairRow:
    """A page and its sample as a manifest row, its images under ``pages/`` and ``refs/``."""
    return PagePairRow(
        id=target.id,
        target=f"{PAGES}/{target.id}.jpg",
        reference=f"{REFS}/{reference.id}.jpg",
        size=target.size,
        volume=target.volume,
        prompts=target.prompts,
        tokens=target.tokens,
    )


def eval_row(target: PageRecord, reference: PageRecord) -> ReferencedCrop:
    """A held-out page as ``diffu_page.cli.evaluate`` draws it: its look form's caption and lines, its sample beside
    it with the sample's line boxes."""
    return ReferencedCrop(
        file_name=f"{target.id}.jpg",
        caption=target.caption,
        width=target.size[0],
        height=target.size[1],
        lines=target.lines,
        reference=f"{REFS}/{reference.id}.jpg",
        reference_lines=[line.bbox for line in reference.lines],
    )


def write_eval_set(out: Path, pairs: list[tuple[PageRecord, PageRecord]], count: int) -> int:
    """``count`` of the held-out ``pairs``, spread over them, as ``out/val_eval``, their images hard-linked there.

    Returns:
        How many pages it holds.
    """
    spread = pairs[:: max(1, len(pairs) // count)][:count] if count else []
    directory = out / VAL_EVAL
    (directory / REFS).mkdir(parents=True, exist_ok=True)
    for target, reference in spread:
        link(out / PAGES / f"{target.id}.jpg", directory / f"{target.id}.jpg")
        link(out / REFS / f"{reference.id}.jpg", directory / REFS / f"{reference.id}.jpg")
    write_jsonl(directory / "metadata.jsonl", [eval_row(target, reference) for target, reference in spread])
    return len(spread)


def build(out: Path, settings: BuildSettings, *, workers: int) -> BuildRecord:
    """The page pairs of ``settings``' scans built under ``out`` (the module's docstring), resuming a stopped build.

    Returns:
        The build's record, also written to ``out/build.json``.

    Raises:
        ValueError: If ``out`` holds a build cut or prompted by other settings.
    """
    scan_record = out / SCAN_RECORD
    if scan_record.exists():
        earlier = ScanSettings.model_validate_json(scan_record.read_text(encoding="utf-8"))
        if earlier != settings.scan:
            raise ValueError(f"{out} holds a build of {earlier}; resume it with those settings or give a new dir")
    for directory in (SCANNED, PAGES, REFS):
        (out / directory).mkdir(parents=True, exist_ok=True)
    scan_record.write_text(settings.scan.model_dump_json(indent=1), encoding="utf-8")

    root = Path(settings.scan.gt_root)
    files = page_files(root, settings.collections, max_scans=settings.max_scans, seed=settings.seed)
    tasks = [ScanTask(xml=xml, out=out, settings=settings.scan) for xml in files]
    results: list[ScanResult] = []
    for k, result in enumerate(scan_all(tasks, workers), start=1):
        results.append(result)
        if k % 500 == 0:
            console.print(f"{k} of {len(tasks)} scans read")

    pages = sorted((page for result in results for page in result.pages), key=lambda page: page.id)
    volumes: dict[str, list[PageRecord]] = defaultdict(list)
    for page in pages:
        volumes[page.volume].append(page)
    held_out = held_out_volumes(volumes, settings.val_volumes, settings.seed)
    rng = random.Random(settings.seed)
    train: list[tuple[PageRecord, PageRecord]] = []
    val: list[tuple[PageRecord, PageRecord]] = []
    too_long = no_partner = 0
    for page in pages:
        if page.too_long:
            too_long += 1
            continue
        partner = pick_partner(page, volumes[page.volume], rng)
        if partner is None:
            no_partner += 1
            continue
        (val if page.volume in held_out else train).append((page, partner))
    write_jsonl(out / "train.jsonl", [pair_row(*pair) for pair in train])
    write_jsonl(out / "val.jsonl", [pair_row(*pair) for pair in val])
    eval_pages = write_eval_set(out, val, settings.eval_pages)

    problems = [result.problem for result in results]
    record = BuildRecord(
        scan=settings.scan,
        collections=settings.collections,
        max_scans=settings.max_scans,
        val_volumes=settings.val_volumes,
        eval_pages=settings.eval_pages,
        seed=settings.seed,
        held_out_volumes=sorted(held_out),
        counts=BuildCounts(
            scans=len(results),
            no_scan=problems.count(ScanProblem.NO_SCAN),
            no_text=problems.count(ScanProblem.NO_TEXT),
            spread=problems.count(ScanProblem.SPREAD),
            failed=problems.count(ScanProblem.FAILED),
            pages=len(pages),
            too_long=too_long,
            no_partner=no_partner,
            train_pairs=len(train),
            val_pairs=len(val),
            eval_pages=eval_pages,
        ),
        failed={result.xml: result.error for result in results if result.problem is ScanProblem.FAILED},
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
    megapixels: Annotated[float, typer.Option(min=0.25, max=4.0, help="a page's size, as drawing plans it")] = 1.0,
    val_volumes: Annotated[float, typer.Option(min=0.0, max=1.0, help="the share of volumes held out")] = 0.03,
    eval_pages: Annotated[int, typer.Option(min=0, help="held-out pages in val_eval/ for diffu-evaluate")] = 24,
    workers: Annotated[int, typer.Option(min=1, help="processes reading the scans: fewer beside a training")] = max(
        1, (os.cpu_count() or 2) // 2
    ),
    seed: Annotated[int, typer.Option(help="picks the scans read, the held-out volumes and each page's sample")] = 0,
) -> None:
    """Build page pairs for MOSS-Image's reference training: whole single pages, each with a sample page."""
    root = gt_root.expanduser()
    settings = BuildSettings(
        scan=ScanSettings(gt_root=str(root), model=str(model), pixels=round(megapixels * 1_000_000)),
        collections=collections or [],
        max_scans=max_scans,
        val_volumes=val_volumes,
        eval_pages=eval_pages,
        seed=seed,
    )
    try:
        record = build(out, settings, workers=workers)
    except ValueError as error:  # a dir holding another build: the arguments' fault
        raise typer.BadParameter(str(error), param_hint="--out") from error
    console.print(f"{record.counts.model_dump_json()} -> {out}")
    console.print(f"held out: {', '.join(record.held_out_volumes)}")


if __name__ == "__main__":
    app()
