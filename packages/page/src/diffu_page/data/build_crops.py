"""FLUX.3 Klein LoRA training crops with layout captions, from the HTR ground truth (PAGE XML pages).

Archival spreads are ~5000 px wide with 60-120 lines, so a whole page at FLUX's ~1 MP shrinks the letters
to a few pixels. Each crop instead covers consecutive WHOLE lines of one text column (a text region plus
the margin notes, headings and page numbers beside, above or below it), scaled so the line spacing is
about ``--line-spacing`` px and saved at the exact size of a NeMo bucket. Its caption follows BFL's
layout-prompt guide: the collection, the volume's years and the scan's look (the style handles), every
line cited as ``<Sv_Text_i>`` with special lines placed in words, then FLUX.3's layout JSON
(``[y_min, x_min, y_max, x_max]`` on 0-1000, y first) quoting each line's transcription.

Input: the Transkribus-export layout of the GT folders, ``<root>/<collection>/<volume>/<volume>/`` with the
scans and a ``page/`` folder of PAGE XML named like them. Output: an imagefolder dataset per split
(``<out>/<split>/<id>.jpg`` + ``metadata.jsonl``, one ``CropRecord`` per crop), loadable with
``datasets.load_dataset("<out>/<split>")``.

    python -m diffu_page.data.build_crops \
        --gt-root ~/Desktop/data/GT-Lejonet_HTR_202602 --max-pages 40 --out-dir data_flux3/layout_crops_v2
"""

from __future__ import annotations

import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, TextIO

import typer
from PIL import Image
from pydantic import BaseModel

from diffu_page.captions.look import scan_look
from diffu_page.captions.records import CropRecord, crop_record
from diffu_page.crops.geometry import CropSettings, Line, Region, crops_for_page
from diffu_page.layout.page_xml import iter_local, local_name, read_page_line, reading_order_index, region_role
from diffu_page.output import console, err_console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

SPLITS = ("train", "val")
SCAN_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class Page(BaseModel):
    """A ground-truth page with a scan and transcribed text."""

    collection: str
    path: Path  # the PAGE XML, relative to the GT root
    scan: Path
    regions: list[Region]

    @property
    def id(self) -> str:
        """A stable id from the page's path."""
        return hashlib.sha1(str(self.path).encode()).hexdigest()[:12]


def parse_page(xml_path: Path) -> list[Region]:
    """A PAGE XML page's text regions in reading order, each with its transcribed lines from top to bottom.

    Raises:
        ValueError: If the file holds no PAGE ``Page`` element.
    """
    pages = iter_local(ET.parse(xml_path).getroot(), "Page")
    if not pages:
        raise ValueError(f"{xml_path} holds no PAGE Page element")
    page = pages[0]
    ordered: list[tuple[int, Region]] = []
    for position, element in enumerate(iter_local(page, "TextRegion")):
        role = region_role(element)
        read = (read_page_line(child) for child in element if local_name(child.tag) == "TextLine")
        lines = sorted(
            (
                Line(polygon=line.polygon, baseline_y=line.baseline_y, text=line.text, role=role)
                for line in read
                if line
            ),
            key=lambda line: line.baseline_y,
        )
        if lines:
            ordered.append((reading_order_index(element, position), Region(lines=lines)))
    return [region for _, region in sorted(ordered, key=lambda item: item[0])]


def sample_pages(collection_dir: Path, count: int, rng: random.Random) -> tuple[list[Page], dict[str, str]]:
    """Up to ``count`` random pages of a collection with a scan and transcribed text, and the pages that could not be
    read, by path, with their error."""
    xml_paths = sorted(p for p in collection_dir.rglob("page/*.xml") if not p.name.startswith("._"))  # "._*": macOS
    rng.shuffle(xml_paths)
    pages: list[Page] = []
    failed: dict[str, str] = {}
    for xml_path in xml_paths:
        if len(pages) == count:
            break
        try:
            page = _load_page(collection_dir, xml_path)
        except Exception as error:  # one unreadable page (a truncated XML) must not end the build; it is reported
            failed[str(xml_path.relative_to(collection_dir.parent))] = f"{type(error).__name__}: {error}"
            continue
        if page is not None:
            pages.append(page)
    return pages, failed


def _load_page(collection_dir: Path, xml_path: Path) -> Page | None:
    scan = _find_scan(xml_path)
    if scan is None:
        return None
    regions = parse_page(xml_path)
    if not regions:
        return None
    return Page(
        collection=collection_dir.name,
        path=xml_path.relative_to(collection_dir.parent),
        scan=scan,
        regions=regions,
    )


def _find_scan(xml_path: Path) -> Path | None:
    """The scan a PAGE XML in ``<volume>/page/`` describes: the same stem, one folder up."""
    candidates = (xml_path.parent.parent / f"{xml_path.stem}{suffix}" for suffix in SCAN_SUFFIXES)
    return next((path for path in candidates if path.is_file()), None)


def split_of(page: Page, val_frac: float) -> str:
    """The page's split, by its hash, so a page's crops never straddle train and val."""
    return "val" if int(page.id, 16) % 1000 < val_frac * 1000 else "train"


def write_crops(page: Page, split_dir: Path, settings: CropSettings) -> list[CropRecord]:
    """Cut a page into crops, save them as JPEGs in ``split_dir`` and return their metadata records. Every crop is cut
    and captioned before any is saved, so a page that fails leaves no tile without its record."""
    with Image.open(page.scan) as img:
        scan = img.convert("RGB")
    tiles: list[Image.Image] = []
    records: list[CropRecord] = []
    for k, crop in enumerate(crops_for_page(page.regions, scan.size, settings)):
        left, top, right, bottom = (round(v) for v in crop.rect)
        tile = scan.crop((left, top, right, bottom)).resize(crop.size, Image.Resampling.LANCZOS)
        name = f"{page.collection}_{page.id}_{k:02d}.jpg"
        tiles.append(tile)
        records.append(
            crop_record(crop, file_name=name, collection=page.collection, page=page.path, look=scan_look(tile))
        )
    for tile, record in zip(tiles, records, strict=True):
        tile.save(split_dir / record.file_name, quality=95)
    return records


def _create_metadata(split_dir: Path) -> TextIO:
    split_dir.mkdir(parents=True, exist_ok=True)
    return open(split_dir / "metadata.jsonl", "w", encoding="utf-8")


def build(
    gt_root: Path,
    collections: list[str],
    out: Path,
    *,
    max_pages: int,
    val_frac: float,
    seed: int,
    settings: CropSettings,
) -> dict[str, str]:
    """Cut ``max_pages`` random pages of each collection into crops, written under ``out/<split>``.

    Returns:
        The pages that failed, by path, with their error; the others are written. They are also written to
        ``out/failed_pages.json``, even when the build stops early.
    """
    rng = random.Random(seed)
    totals: Counter[str] = Counter()
    failed: dict[str, str] = {}
    out.mkdir(parents=True, exist_ok=True)
    try:
        with ExitStack() as stack:
            metadata = {split: stack.enter_context(_create_metadata(out / split)) for split in SPLITS}
            for collection in collections:
                pages, unreadable = sample_pages(gt_root / collection, max_pages, rng)
                failed.update(unreadable)
                for page in pages:
                    split = split_of(page, val_frac)
                    try:
                        records = write_crops(page, out / split, settings)
                    except Exception as error:  # one unreadable page must not end the build; it is reported
                        failed[str(page.path)] = f"{type(error).__name__}: {error}"
                        continue
                    metadata[split].writelines(f"{record.model_dump_json()}\n" for record in records)
                    totals.update(pages=1, crops=len(records), lines=sum(len(record.lines) for record in records))
                console.print(f"{collection}: {dict(totals)}")
    finally:
        (out / "failed_pages.json").write_text(json.dumps(failed, indent=1, ensure_ascii=False), encoding="utf-8")
    return failed


@app.command()
def main(
    gt_root: Annotated[Path, typer.Option(help="the GT folder, one subfolder per collection")],
    out_dir: Annotated[Path, typer.Option(help="where train/ and val/ go")],
    collections: Annotated[
        list[str] | None, typer.Option("--collection", help="a collection to use (repeat it; default: all)")
    ] = None,
    max_pages: Annotated[int, typer.Option(help="pages sampled per collection")] = 40,
    val_frac: Annotated[float, typer.Option(help="pages held out, by page hash")] = 0.05,
    seed: int = 0,
    line_spacing: Annotated[float, typer.Option(help="target line pitch in px")] = CropSettings().line_spacing,
    min_line_spacing: Annotated[
        float, typer.Option(help="skip crops scaled below this pitch")
    ] = CropSettings().min_line_spacing,
    min_merge_spacing: Annotated[
        float, typer.Option(help="smallest pitch a margin note may shrink its column to")
    ] = CropSettings().min_merge_spacing,
    min_width: Annotated[
        int, typer.Option(help="narrower regions are only cropped together with a wide one")
    ] = CropSettings().min_width,
    min_lines: int = CropSettings().min_lines,
    overlap_lines: Annotated[
        int, typer.Option(help="lines shared by consecutive windows of a column")
    ] = CropSettings().overlap_lines,
) -> None:
    """Cut GT pages into FLUX.3 training crops with layout captions."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise typer.BadParameter(
            f"{out_dir} is not empty: an earlier build's crops would stay beside this one's", param_hint="--out-dir"
        )
    Image.MAX_IMAGE_PIXELS = None  # archival spreads exceed PIL's decompression-bomb default
    root = gt_root.expanduser()
    settings = CropSettings(
        line_spacing=line_spacing,
        min_line_spacing=min_line_spacing,
        min_merge_spacing=min_merge_spacing,
        min_width=min_width,
        min_lines=min_lines,
        overlap_lines=overlap_lines,
    )
    names = collections or sorted(p.name for p in root.iterdir() if p.is_dir())
    failed = build(root, names, out_dir, max_pages=max_pages, val_frac=val_frac, seed=seed, settings=settings)
    if failed:
        err_console.print(f"{len(failed)} pages failed, see {out_dir / 'failed_pages.json'}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
