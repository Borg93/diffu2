"""Region crops for a cache of page pairs: each entry's target given a crop of another text region of its own page for
each of its handwritten text regions (:func:`diffu_page.draw.line_samples.plan_region_crops`), the manifest that
``diffu-moss-cache-region-crops`` encodes in place of the entries' whole samples. CPU only.

    python -m diffu_page.data.region_crop_pairs --cache data_pages/cache_above_2mp \\
        --out data_pages/region_crops_over --place over

The cache is a finished ``diffu-moss-cache-pairs`` cache of a page-pair set (``diffu_page.data.page_pairs``) and the
manifest it was made from (its record's, or ``--pairs``: a set's ``train.jsonl``, or the sibling rows of
``train_siblings.jsonl``, each window with its sibling window as its sample), which lies in the set's directory, beside
its ``scanned/`` records and its ``pages/`` and ``windows/`` images. One page is almost always one hand, so the crops
are the target's scribe for certain. A window's crops come from its sibling window: the window of its page its row
names as its sample, else another window of its page (picked by ``--seed``) sharing none of its lines; a whole page's
from its own other regions, never a region's own. Each source is changed by the entry's look change as its sample was
(``diffu_moss.train.cache.styled``) before its crops are cut, and the crops are written as PNG files.

An entry is left out, and counted by its reason, when its form takes no sample, it took a hand change (its target's
lines moved off their recorded boxes), its window has no sibling sharing none of its lines, its target has no
handwritten line, or no region of it has another region to be given. Written under ``--out``: ``crops/<entry>_<k>.png``,
``region_crops.jsonl`` (a row an entry served: :class:`diffu_moss.train.region_crops.RegionCropRow`) and, last,
``region_crops.json`` (:class:`diffu_moss.train.region_crops.RegionCropBuild`).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from diffu_moss.train.cache import CacheEntry, read_entries, read_record, styled
from diffu_moss.train.pairs import PagePairRow
from diffu_moss.train.region_crops import (
    CROP_IMAGES,
    REGION_CROP_BUILD,
    REGION_CROPS,
    RegionCropBuild,
    RegionCropFile,
    RegionCropRow,
)
from diffu_moss.transformer import CropPlace
from diffu_page.captions.prompt import LineRecord
from diffu_page.data.evalset import read_jsonl
from diffu_page.data.page_pairs import PAGES, WINDOWS, PageRecord, Planned, WindowRecord, record_reader
from diffu_page.draw.line_samples import cut_line_crops, plan_region_crops
from diffu_page.output import console

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class Skipped(StrEnum):
    """Why an entry is given no crops."""

    NO_SAMPLE = "no sample"  # its form is drawn without one
    HAND_CHANGE = "hand change"  # its target's lines no longer lie in their recorded boxes
    NO_SIBLING = "no sibling"  # its window's page has no other window sharing none of its lines
    NO_HANDWRITING = "no handwriting"  # its target has no handwritten line
    NO_OTHER_REGION = "no other region"  # none of its regions has a source region holding none of its lines


class RegionCropSettings(BaseModel):
    """Where the crops lie (over their region's box, or beside the page in its rows), their largest size (px) and the
    seed that orders a source's regions and picks a window's sibling where its row names none."""

    model_config = ConfigDict(frozen=True)

    place: CropPlace = "over"
    max_width: int = Field(default=512, ge=16)
    max_height: int = Field(default=512, ge=16)
    seed: int = 0


class Source(BaseModel):
    """The image a target's crops are cut from and its lines (boxes 0-1000 of it, y first)."""

    model_config = ConfigDict(frozen=True)

    image: Path
    size: tuple[int, int]
    lines: list[LineRecord]


def sibling_of(row: PagePairRow, page: PageRecord, target: WindowRecord, seed: int) -> WindowRecord | None:
    """The window of ``page`` a window row's crops are cut from: the one its row names as its sample, else one of the
    page's other windows picked by ``seed``; only a window sharing none of ``target``'s lines, None when there is none."""
    others = [window for window in page.windows if window.id != target.id and window.texts().isdisjoint(target.texts())]
    named = row.sample or Path(row.reference).stem
    return next(
        (window for window in others if window.id == named),
        random.Random(f"{seed}/{row.id}").choice(others) if others else None,
    )


def row_source(
    row: PagePairRow, page_of: Callable[[str], PageRecord], pairs_dir: Path, seed: int
) -> tuple[Planned, Source | None]:
    """A row's target as planned, and the source its crops are cut from: a whole page's own image, a window's
    sibling's (:func:`sibling_of`; None when it has none), in the set at ``pairs_dir``."""
    if row.window is None:
        page = page_of(row.id)
        return page, Source(image=pairs_dir / PAGES / f"{page.id}.jpg", size=page.size, lines=page.lines)
    page = page_of(row.window.page)
    target = next(window for window in page.windows if window.id == row.id)
    sibling = sibling_of(row, page, target, seed)
    if sibling is None:
        return target, None
    return target, Source(image=pairs_dir / WINDOWS / f"{sibling.id}.jpg", size=sibling.size, lines=sibling.lines)


def entry_crops(
    entry: CacheEntry,
    rows: Mapping[str, PagePairRow],
    page_of: Callable[[str], PageRecord],
    pairs_dir: Path,
    out: Path,
    settings: RegionCropSettings,
) -> RegionCropRow | Skipped:
    """``entry``'s crops, their images written under ``out``, or why it is given none.

    Raises:
        ValueError: If the entry's pair has no row in ``rows``, or its target is not the size its record gives.
    """
    if not entry.variant.takes_reference:
        return Skipped.NO_SAMPLE
    if entry.hand is not None:
        return Skipped.HAND_CHANGE
    if entry.pair not in rows:
        raise ValueError(f"the cache's pair {entry.pair} has no row in the manifest: give the one it was made from")
    target, source = row_source(rows[entry.pair], page_of, pairs_dir, settings.seed)
    if entry.target_size != target.size:
        raise ValueError(f"{entry.pair} was cached at {entry.target_size} px, its set's record gives {target.size}")
    if source is None:
        return Skipped.NO_SIBLING
    try:
        crops = plan_region_crops(
            source.size,
            source.lines,
            target.size,
            target.lines,
            place=settings.place,
            max_width=settings.max_width,
            max_height=settings.max_height,
            seed=settings.seed,
        )
    except ValueError:  # the target has no handwritten line
        return Skipped.NO_HANDWRITING
    if not crops:
        return Skipped.NO_OTHER_REGION
    with Image.open(source.image) as image:
        images = cut_line_crops(styled(image, restyle=entry.restyle), crops)
    files = []
    for index, (crop, cut) in enumerate(zip(crops, images, strict=True)):
        name = f"{CROP_IMAGES}/{Path(entry.file).stem}_{index}.png"
        cut.save(out / name)
        files.append(
            RegionCropFile(file=name, source=crop.source, target=crop.target, size=crop.size, offset=crop.offset)
        )
    return RegionCropRow(entry=entry.file, pair=entry.pair, crops=files)


def build_region_crops(
    cache: Path, out: Path, settings: RegionCropSettings, *, pairs: Path | None = None
) -> RegionCropBuild:
    """The crops of the entries of the finished cache ``cache`` (the module's docstring), from the page-pair manifest
    ``pairs`` (default: the one ``cache`` was made from) and its set's records, written under ``out``.

    Returns:
        The build's record, also written to ``out/region_crops.json``.

    Raises:
        ValueError: If ``out`` holds a build already, ``cache`` is not finished, or an entry has no row in the manifest
            or was cached at another size than its record gives.
    """
    if (out / REGION_CROP_BUILD).exists():
        raise ValueError(f"{out} holds region crops already; give a new directory")
    manifest = Path(read_record(cache).pairs) if pairs is None else pairs
    rows = {row.id: row for row in read_jsonl(manifest, PagePairRow)}
    page_of = record_reader(manifest.parent)
    entries = read_entries(cache)
    (out / CROP_IMAGES).mkdir(parents=True, exist_ok=True)
    skipped: Counter[str] = Counter()
    served = 0
    with (out / REGION_CROPS).open("w", encoding="utf-8") as written:
        for index, entry in enumerate(entries, start=1):
            crops = entry_crops(entry, rows, page_of, manifest.parent, out, settings)
            if isinstance(crops, Skipped):
                skipped[crops.value] += 1
            else:
                written.write(f"{crops.model_dump_json()}\n")
                served += 1
            if index % 1000 == 0:
                console.print(f"{index} of {len(entries)} entries read, {served} served")
    record = RegionCropBuild(
        cache=str(cache.resolve()),
        pairs=str(manifest.resolve()),
        place=settings.place,
        max_width=settings.max_width,
        max_height=settings.max_height,
        seed=settings.seed,
        entries_read=len(entries),
        served=served,
        skipped=dict(sorted(skipped.items())),
    )
    record.write(out / REGION_CROP_BUILD)
    return record


@app.command()
def main(
    cache: Annotated[Path, typer.Option(help="a finished diffu-moss-cache-pairs cache of a page-pair set")],
    out: Annotated[Path, typer.Option(help="the crops' directory (a new one)")],
    pairs: Annotated[
        Path | None,
        typer.Option(
            help="the page-pair manifest the cache was made from, in its set's directory (default: its record's)"
        ),
    ] = None,
    place: Annotated[
        CropPlace, typer.Option(help="where a crop lies: over its region's box, or beside the page in the box's rows")
    ] = "over",
    max_width: Annotated[int, typer.Option(min=16, help="a crop's widest, px (cut, never shrunk)")] = 512,
    max_height: Annotated[int, typer.Option(min=16, help="a crop's tallest, px (cut, never shrunk)")] = 512,
    seed: Annotated[int, typer.Option(help="orders a source's regions; picks a sibling where a row names none")] = 0,
) -> None:
    """Give each entry of a page-pair cache a crop of another text region of its page for each of its regions."""
    try:
        settings = RegionCropSettings(place=place, max_width=max_width, max_height=max_height, seed=seed)
        record = build_region_crops(cache, out, settings, pairs=pairs)
    except ValueError as error:  # a used dir, another cache's manifest: the arguments' fault
        raise typer.BadParameter(str(error)) from error
    console.print(f"{record.served} of {record.entries_read} entries served, left out: {record.skipped} -> {out}")


if __name__ == "__main__":
    app()
