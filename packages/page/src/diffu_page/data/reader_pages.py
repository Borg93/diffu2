"""The pages a page reader's fine-tuning set is written from (:mod:`diffu_page.cli.reader_set`), each with what the set's
sidecar records of it (:class:`SampleMeta`): a diffu-generate run's images with their elements files
(:func:`run_scans`), or PAGE or ALTO files with their scans (:func:`xml_scans`), the GT's or a drawn run's
(``diffu-run-export``'s ``<id>.page.xml`` and ``<id>.alto.xml`` beside ``<id>.jpg``).

The files are those below a folder (every ``*.xml`` that is a PAGE or an ALTO file: a volume's METS is passed over, as
are macOS's ``._*`` files) or those a list names, one path a line, read from the list's folder. A file's scan is the
image it names (PAGE ``Page/@imageFilename``, ALTO ``sourceImageInformation/fileName``), else one named as the file
(less ``.xml``, ``.page.xml`` or ``.alto.xml``), beside it or one folder up: the Transkribus export keeps a volume's
PAGE files in ``page/`` and its ALTO files in ``alto/``, beside its scans. A scan both a PAGE and an ALTO file describe
is read from its PAGE file, which holds its regions' types and its tables (an ALTO file, as read, holds neither: every
region is body text). A file diffu_page wrote (its PAGE ``Creator``, its ALTO ``softwareName``) is a synthetic page,
any other a real one.

Each scan is read as its layout (:func:`load_xml_layout`) and split into its pages (:func:`scan_pages`: an opening's
two, each in its own pixels, else the scan whole), each page made into the readers' elements (:func:`layout_elements`:
a region's type its kind, an untyped region body text). A page's id is that of the MOSS page-pair sets
(:func:`page_id`: the digest of its file's path below the folder given, or the list's folder, and its side of an
opening), so given the GT's root the ids are theirs. Its collection is the nearest folder above its file named as the
GT names one (``GT-...``), its volume the folder its file's ``page/`` or ``alto/`` folder is in ("" where there is
none, as for a drawn run's pages).

Left out, and recorded (:class:`Skipped`): the GT's second export of a collection (:data:`DUPLICATE_EXPORTS`); a file
without its scan, one that cannot be read and one whose scan is not the size it says; a scan without a transcribed
line, and one wider than tall that its lines cannot split (:func:`is_unsplit_spread`), as no spread is trained on; and
the pages an exclusion list names (:func:`read_exclusions`, :func:`page_names`). A scan's split (:func:`gt_split`) is
the GT's own: a scan a test list of its volume names is a test page, never a training one.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from enum import StrEnum
from functools import cache
from itertools import chain
from pathlib import Path, PurePosixPath

from PIL import Image
from pydantic import BaseModel

from diffu_page.local_run import describe
from diffu_page.page.labels import ELEMENTS_FILE, LabelFormat, PageElements, layout_elements
from diffu_page.page.layout import (
    LineKind,
    PageSide,
    Production,
    ScanPage,
    is_unsplit_spread,
    load_xml_layout,
    scan_pages,
)
from diffu_page.page.layout.page_xml import SCAN_SUFFIXES, local_name
from diffu_page.page.layout.pages import digest, page_id

DRAWN_BY = "diffu_page"  # who a drawn page's PAGE file names as its Creator, its ALTO file as its softwareName

# The GT's later export of a collection it holds twice, left out, by the export kept: the same scans again (md5-checked),
# its test lists naming others of them (34 against 31, 3 shared), which count for the kept export's scans too
DUPLICATE_EXPORTS: dict[str, str] = {"GT-RA_landskapshandlingar_20260227": "GT-RA_landskapshandlingar_20260220"}
TEST_LISTS = "test_*"  # a volume's test lists in its page/ folder, nested: test_0025 in test_0050 ... in test_0100
COLLECTION_PREFIX = "GT-"  # how the GT names its collections' folders (GT-RA_..., GT-SSA_..., GT-FNA_...)

_FORMATS = {"PcGts": LabelFormat.PAGE, "alto": LabelFormat.ALTO}  # a label file's root element

type PixelRect = tuple[int, int, int, int]  # left, top, right, bottom


class Source(StrEnum):
    """Where a page is from: a real scan, or drawn (synthetic)."""

    REAL = "real"
    SYNTHETIC = "synthetic"


class Split(StrEnum):
    """A sample's split: trained on, or held out to test on."""

    TRAIN = "train"
    TEST = "test"


class SampleMeta(BaseModel):
    """What a set's sidecar records of a sample: its id, whether its page is real or drawn, its collection and volume
    ("" where the input does not give them), its scan's file name, its side of an opening (None: a single page, or an
    opening shown whole) and its split (None: the set is not split); for an element's crop, its kind and its place in
    the page's reading order, and for a line's crop also how the line was made and its place in its element (None
    where the sample is a page)."""

    id: str
    source: Source
    collection: str = ""
    volume: str = ""
    scan: str
    side: PageSide | None = None
    split: Split | None = None
    kind: LineKind | None = None
    production: Production | None = None
    element: int | None = None
    line: int | None = None


class ReaderPage(BaseModel):
    """A page to write samples of: its elements (its own pixels), the image it is on (an absolute path), where on that
    image it is (None: all of it) and its meta, whose id is the page's."""

    elements: PageElements
    image: Path
    rect: PixelRect | None = None
    meta: SampleMeta


class ReaderScan(BaseModel):
    """A scan's pages to write, in reading order (an opening's left page first), the scan's id and the file its labels
    are read from."""

    id: str
    labels: str
    pages: list[ReaderPage]


class SkipReason(StrEnum):
    """Why a file, scan or page is left out of a set."""

    DUPLICATE = "duplicate"  # the GT's second export of a collection
    NO_SCAN = "no-scan"
    UNREADABLE = "unreadable"
    SIZE = "size"  # the scan is not the size its file gives
    NO_TEXT = "no-text"
    SPREAD = "spread"  # wider than tall, its lines cannot split it
    EXCLUDED = "excluded"
    EMPTY_LINE = "empty-line"  # a line with no text to read
    UNCLEAR_LINE = "unclear-line"  # a drawn line that read back badly (tagged unclear)
    SMALL_LINE = "small-line"  # a line's box too narrow or too low to hold writing
    CAPPED = "capped"  # its collection already holds the samples a set takes of one


class Skipped(BaseModel):
    """A file, scan, page or line left out of a set: its label file, its side of an opening (a page or line alone),
    why, the error that made it unreadable and, for a line, the id its sample would have had."""

    labels: str
    side: PageSide | None = None
    reason: SkipReason
    error: str = ""
    id: str = ""


class LabelFile(BaseModel):
    """A PAGE or ALTO file to read a scan's pages from: its path, format and scan (absolute) and whether it is drawn."""

    path: Path
    format: LabelFormat
    scan: Path
    source: Source


def open_scan(path: Path) -> Image.Image:
    """The image at ``path``, opened (read when first used): archival spreads pass PIL's decompression-bomb limit."""
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(path)


def run_scans(run: Path) -> list[ReaderScan]:
    """Each image of a diffu-generate run drawn with ``--reader-labels``, with its elements (``<name>.elements.json``),
    as a scan of one synthetic page whose id is the image's stem, in the order of their files' names.

    Raises:
        ValueError: If the run holds no elements files, or one names an image the run does not hold.
    """
    scans: list[ReaderScan] = []
    for path in sorted(run.glob(f"*{ELEMENTS_FILE}")):
        elements = PageElements.model_validate_json(path.read_text(encoding="utf-8"))
        image = (run / elements.image).resolve()
        if not elements.image or not image.is_file():
            raise ValueError(f"{path.name} labels the image {elements.image!r}, which the run does not hold")
        meta = SampleMeta(id=image.stem, source=Source.SYNTHETIC, scan=image.name)
        page = ReaderPage(elements=elements, image=image, meta=meta)
        scans.append(ReaderScan(id=image.stem, labels=str(path), pages=[page]))
    if not scans:
        raise ValueError(f"{run} holds no {ELEMENTS_FILE} files: draw it with diffu-generate --reader-labels")
    return scans


class _Header(BaseModel):
    """What a label file says before its page: its format, its scan's file name (if it names one) and its maker."""

    format: LabelFormat
    image: str | None
    creator: str


def _header(path: Path) -> _Header | None:
    """The header of the PAGE or ALTO file at ``path``, read up to its page; None for any other XML.

    Raises:
        ET.ParseError: If the file is not well-formed XML.
    """
    image: str | None = None
    creator = ""
    with path.open("rb") as source:
        events = ET.iterparse(source, events=("start", "end"))
        _, root = next(events)
        label_format = _FORMATS.get(local_name(root.tag))
        if label_format is None:
            return None
        for event, element in events:
            name = local_name(element.tag)
            if event == "start" and name in ("Page", "Layout"):  # PAGE's page, ALTO's layout: the header has ended
                image = image or element.get("imageFilename")
                break
            if event == "end" and name in ("Creator", "softwareName") and not creator:
                creator = (element.text or "").strip()
            if event == "end" and name == "fileName":
                image = (element.text or "").strip() or None
    return _Header(format=label_format, image=Path(image).name if image else None, creator=creator)


def _stem(path: Path) -> str:
    """A label file's name less ``.xml``, and less ``.page`` or ``.alto`` before it (``<id>.page.xml``: ``<id>``)."""
    stem = path.stem
    return stem.rsplit(".", 1)[0] if stem.endswith((".page", ".alto")) else stem


def _scan(path: Path, image: str | None) -> Path | None:
    """The scan a label file describes: the image it names, else one named as the file, beside it or one folder up."""
    names = ([image] if image else []) + [f"{_stem(path)}{suffix}" for suffix in SCAN_SUFFIXES]
    candidates = (folder / name for folder in (path.parent, path.parent.parent) for name in names)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _listed(source: Path) -> list[Path]:
    """The files a folder holds below it (``*.xml``, but macOS's ``._*``) or a list names (one path a line, read from
    its folder; blank lines and ``#`` comments passed over), absolute and in path order."""
    if source.is_dir():
        return sorted(path for path in source.rglob("*.xml") if not path.name.startswith("._"))
    lines = (line.strip() for line in source.read_text(encoding="utf-8").splitlines())
    return sorted(source.parent / line for line in lines if line and not line.startswith("#"))


def label_files(source: Path) -> tuple[list[LabelFile], list[Skipped]]:
    """The label files ``source`` (an absolute folder or list) gives, one a scan (its PAGE file where both a PAGE and an
    ALTO file describe it) in path order, and the files left out: the GT's second export of a collection, a file that
    cannot be read, one without its scan."""
    kept: dict[Path, LabelFile] = {}
    skipped: list[Skipped] = []
    for path in _listed(source):
        if DUPLICATE_EXPORTS.keys() & set(path.parts):
            skipped.append(Skipped(labels=str(path), reason=SkipReason.DUPLICATE))
            continue
        try:
            header = _header(path)
        except (ET.ParseError, OSError) as error:
            skipped.append(Skipped(labels=str(path), reason=SkipReason.UNREADABLE, error=describe(error)))
            continue
        if header is None:  # another kind of XML file
            continue
        scan = _scan(path, header.image)
        if scan is None:
            skipped.append(Skipped(labels=str(path), reason=SkipReason.NO_SCAN))
            continue
        source_kind = Source.SYNTHETIC if header.creator == DRAWN_BY else Source.REAL
        label = LabelFile(path=path, format=header.format, scan=scan, source=source_kind)
        known = kept.get(scan)
        if known is None or (known.format is LabelFormat.ALTO and label.format is LabelFormat.PAGE):
            kept[scan] = label
    return sorted(kept.values(), key=lambda label: label.path), skipped


@cache
def _test_names(folder: Path) -> frozenset[str]:
    """The file names the GT's test lists in ``folder`` name (every ``test_*`` file's lines), none where it has none."""
    return frozenset(
        name.strip()
        for listing in folder.glob(TEST_LISTS)
        if listing.is_file()
        for name in listing.read_text(encoding="utf-8").splitlines()
        if name.strip()
    )


def gt_split(path: Path) -> Split:
    """A scan's split by the GT's own test lists, given its label file (an absolute path): a test page when a list in
    its volume's ``page/`` folder names its file (the union of its lists: they are nested), or a list of the same folder
    in the export the GT holds of it again (:data:`DUPLICATE_EXPORTS`), else a training page."""
    folders = [path.parent, path.parent.parent / "page"]
    for left_out, kept in DUPLICATE_EXPORTS.items():
        if kept in path.parts:
            twin = Path(*(left_out if part == kept else part for part in path.parts))
            folders += [twin.parent, twin.parent.parent / "page"]
    return Split.TEST if any(path.name in _test_names(folder) for folder in folders) else Split.TRAIN


def unsplit(_: Path) -> None:
    """No split: a set is written whole."""
    return None


def read_exclusions(path: Path) -> frozenset[str]:
    """The pages a list names to leave out, one a line (blank lines and ``#`` comments passed over): each a page's id or
    a path, as :func:`page_names` gives them."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return frozenset(PurePosixPath(line).as_posix() for line in lines if line and not line.startswith("#"))


def _path_names(path: Path) -> set[str]:
    """An absolute ``path`` whole and by its last parts (``.../page/x.xml``: ``page/x.xml``, ``x.xml``)."""
    parts = path.parts[1:]
    return {path.as_posix()} | {PurePosixPath(*parts[k:]).as_posix() for k in range(len(parts))}


def page_names(label: LabelFile, side: PageSide | None) -> set[str]:
    """Every name an exclusion list may give a page by: the path of its label file or of its scan, whole or by its last
    parts (a GT page by its path below the GT's root, as the training crops record it), and its id in a page-pair set,
    the digest of its label file's path below any folder above it, with its side, or without it for all the scan's
    pages."""
    paths = _path_names(label.path)
    scan_ids = {digest(name) for name in paths}
    page_ids = {f"{scan_id}_{side}" for scan_id in scan_ids} if side else set()
    return paths | _path_names(label.scan) | scan_ids | page_ids


def _pixels(page: ScanPage) -> PixelRect | None:
    """Where a page of an opening is on its scan, in whole pixels; None for a scan's single page."""
    if page.side is None:
        return None
    left, top, right, bottom = (round(value) for value in page.rect)
    return left, top, right, bottom


def _collection(path: Path) -> str:
    """The GT collection a label file is of: the nearest folder above it named as the GT names them (``GT-...``)."""
    return next((folder.name for folder in path.parents if folder.name.startswith(COLLECTION_PREFIX)), "")


def _volume(path: Path) -> str:
    """The volume a label file is of: the folder its ``page/`` or ``alto/`` folder is in, beside the volume's scans."""
    return path.parent.parent.name if path.parent.name in ("page", "alto") else ""


def label_scans(
    label: LabelFile, root: Path, *, split_of: Callable[[Path], Split | None], excluded: frozenset[str]
) -> Iterator[ReaderScan | Skipped]:
    """A label file's scan as its pages to write (none when no page is left), ``root`` the folder its ids are taken
    below, ``split_of`` its split, and what of it is left out: the scan, or a page ``excluded`` names."""
    try:
        layout = load_xml_layout(label.path)
        with open_scan(label.scan) as scan:
            size = scan.size
    except (ValueError, OSError) as error:
        yield Skipped(labels=str(label.path), reason=SkipReason.UNREADABLE, error=describe(error))
        return
    if size != (layout.width, layout.height):
        error = f"{label.scan.name} is {size} px, its file gives {(layout.width, layout.height)}"
        yield Skipped(labels=str(label.path), reason=SkipReason.SIZE, error=error)
        return
    if not layout.lines:
        yield Skipped(labels=str(label.path), reason=SkipReason.NO_TEXT)
        return
    if is_unsplit_spread(layout):
        yield Skipped(labels=str(label.path), reason=SkipReason.SPREAD)
        return
    below = label.path.relative_to(root) if label.path.is_relative_to(root) else label.path
    split = split_of(label.path)
    pages: list[ReaderPage] = []
    for page in scan_pages(layout):
        if page_names(label, page.side) & excluded:
            yield Skipped(labels=str(label.path), side=page.side, reason=SkipReason.EXCLUDED)
            continue
        elements = layout_elements(page.layout, label.scan.name)
        if not elements.elements:  # its lines all lie off the scan
            yield Skipped(labels=str(label.path), side=page.side, reason=SkipReason.NO_TEXT)
            continue
        meta = SampleMeta(
            id=page_id(below, page),
            source=label.source,
            collection=_collection(label.path),
            volume=_volume(label.path),
            scan=label.scan.name,
            side=page.side,
            split=split,
        )
        pages.append(ReaderPage(elements=elements, image=label.scan, rect=_pixels(page), meta=meta))
    if pages:
        yield ReaderScan(id=digest(below.as_posix()), labels=str(label.path), pages=pages)


class Selection(BaseModel):
    """Which collections a set takes (``include``: only these, none named: all; never those of ``exclude``), and the
    order its scans are read in: by path, or shuffled by ``seed`` (so a cap on a collection's samples takes scans
    from all its volumes, not its first ones)."""

    model_config = {"frozen": True}

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    seed: int | None = None

    def keeps(self, collection: str) -> bool:
        """Whether the set takes a scan of ``collection``."""
        return collection not in self.exclude and (not self.include or collection in self.include)


def xml_scans(
    source: Path,
    *,
    split_of: Callable[[Path], Split | None],
    excluded: frozenset[str],
    selection: Selection | None = None,
) -> Iterator[ReaderScan | Skipped]:
    """The scans of the PAGE or ALTO files ``source`` gives (a folder, or a list of them) as their pages to write, each
    read as it is taken, in path order (or ``selection``'s) after the files left out (:func:`label_files`,
    :func:`label_scans`); a file of a collection ``selection`` does not take is passed over, and so is its record of
    being left out.

    Raises:
        ValueError: If ``source`` gives no PAGE or ALTO file (none is read when it gives one), or no file of a
            collection ``selection`` names.
    """
    selection = selection or Selection()
    root = source.resolve() if source.is_dir() else source.resolve().parent
    labels, skipped = label_files(source.resolve())
    if not labels and not skipped:
        raise ValueError(f"{source} gives no PAGE or ALTO file")
    present = {_collection(label.path) for label in labels} | {_collection(Path(item.labels)) for item in skipped}
    if unknown := sorted(set(selection.include) - present):
        raise ValueError(f"{source} holds no file of {', '.join(unknown)}")
    labels = [label for label in labels if selection.keeps(_collection(label.path))]
    skipped = [item for item in skipped if selection.keeps(_collection(Path(item.labels)))]
    if selection.seed is not None:
        random.Random(selection.seed).shuffle(labels)
    scans = (item for label in labels for item in label_scans(label, root, split_of=split_of, excluded=excluded))
    return chain(skipped, scans)
