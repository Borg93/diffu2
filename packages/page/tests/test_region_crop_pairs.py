"""Region crops (diffu_page.data.region_crop_pairs) for a cache of page pairs built from a tiny ground truth: each text
region of an entry's target gets a crop of another region of its own page (a window's: of its sibling window), never of
its own lines, in the entry's look; entries that cannot be served are left out and counted by their reason."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from diffu_moss.train.cache import CacheEntry, CacheSettings, read_entries, write_cache
from diffu_moss.train.pairs import PagePairRow, PromptMix
from diffu_moss.train.region_crops import REGION_CROPS, RegionCropBuild, read_region_crops
from diffu_page.captions.prompt import LineRecord
from diffu_page.data.evalset import read_jsonl
from diffu_page.data.page_pairs import BuildSettings, ScanSettings, build, record_reader
from diffu_page.data.region_crop_pairs import RegionCropSettings, Skipped, build_region_crops
from diffu_page.draw.line_samples import region_of

VOLUME = Path("GT-RA_arjeplogs_tingslag_20260220", "Arjeplogs tingslag (1690-1700)", "Arjeplogs tingslag (1690-1700)")
PAPER, INK = (222, 205, 170), (70, 50, 30)
REGION_LINES = 20  # each page: three text regions of this many lines, one under the other, at a 40 px pitch


def _write_page(pages: Path, name: str) -> None:
    """A PAGE file ``pages/<name>.xml`` of 800 x 1900 px and its scan: two text regions, ``a`` and ``b``, of
    :data:`REGION_LINES` lines each (x 100-700, a 40 px pitch from y 60), each line a dark bar."""
    regions, bars = [], []
    for k, region in enumerate("abc"):
        lines = []
        for i in range(REGION_LINES):
            top = 60 + 40 * (k * REGION_LINES + i)
            box = (100, top, 700, top + 30)
            lines.append(
                f'<TextLine id="{region}{i}"><Coords points="100,{top} 700,{top} 700,{top + 30} 100,{top + 30}"/>'
                f'<Baseline points="100,{top + 20} 700,{top + 20}"/>'
                f"<TextEquiv><Unicode>{name} {region} rad {i} om tinget</Unicode></TextEquiv></TextLine>"
            )
            bars.append(box)
        regions.append(f'<TextRegion id="{region}" type="paragraph">{"".join(lines)}</TextRegion>')
    pages.mkdir(parents=True, exist_ok=True)
    (pages / f"{name}.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        f'<Page imageFilename="{name}.png" imageWidth="800" imageHeight="2500">{"".join(regions)}</Page></PcGts>',
        encoding="utf-8",
    )
    image = Image.new("RGB", (800, 2500), PAPER)
    draw = ImageDraw.Draw(image)
    for box in bars:
        draw.rectangle(box, fill=INK)
    image.save(pages.parent / f"{name}.png")


@pytest.fixture(scope="module")
def page_set(tiny_moss: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A page-pair set of three pages of one volume, up to three windows each, and its sibling rows
    (``train_siblings.jsonl``): each window with another window of its page sharing none of its lines as its sample,
    beside the whole pages."""
    gt = tmp_path_factory.mktemp("gt")
    for name in ("0001", "0002", "0003"):
        _write_page(gt / VOLUME / "page", name)
    out = tmp_path_factory.mktemp("page_set")
    scan = ScanSettings(gt_root=str(gt), model=str(tiny_moss), pixels=192 * 192, windows_per_page=3)
    build(out, BuildSettings(scan=scan, val_volumes=0.0, eval_pages=0), workers=1)
    rows = read_jsonl(out / "train.jsonl", PagePairRow)
    page_of = record_reader(out)
    siblings = [row for row in rows if row.window is None]
    for row in rows:
        if row.window is None:
            continue
        window = next(window for window in page_of(row.window.page).windows if window.id == row.id)
        others = [
            other
            for other in page_of(row.window.page).windows
            if other.id != row.id and other.texts().isdisjoint(window.texts())
        ]
        if others:
            siblings.append(row.model_copy(update={"reference": f"windows/{others[0].id}.jpg"}))
    (out / "train_siblings.jsonl").write_text("".join(f"{row.model_dump_json()}\n" for row in siblings))
    return out


@pytest.fixture(scope="module")
def page_cache(tiny_moss: Path, page_set: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The sibling rows cached in every form of the content mix, a look change for half the sample-form entries and a
    hand change for a third of them."""
    cache = tmp_path_factory.mktemp("page_cache") / "cache"
    settings = CacheSettings(
        mix=PromptMix.CONTENT, every_form=True, restyle_share=0.5, hand_restyle_share=0.34, device="cpu"
    )
    write_cache(tiny_moss, page_set / "train_siblings.jsonl", cache, settings)
    return cache


@pytest.fixture(scope="module")
def built(page_cache: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The cache's region crops, laid over their regions' boxes."""
    out = tmp_path_factory.mktemp("region_crops")
    build_region_crops(page_cache, out, RegionCropSettings())
    return out


def _texts(lines: list[LineRecord]) -> set[str]:
    return {line.text for line in lines}


def _regions(lines: list[LineRecord]) -> dict[str, set[str]]:
    regions: dict[str, set[str]] = {}
    for line in lines:
        regions.setdefault(region_of(line), set()).add(line.text)
    return regions


def _target_and_source(page_set: Path, pair: str) -> tuple[list[LineRecord], list[LineRecord], Path]:
    """A sibling row's target's lines, and its source's lines and image: a window's sibling (its sample), a page's
    own."""
    row = next(row for row in read_jsonl(page_set / "train_siblings.jsonl", PagePairRow) if row.id == pair)
    page_of = record_reader(page_set)
    if row.window is None:
        page = page_of(row.id)
        return page.lines, page.lines, page_set / row.target
    windows = {window.id: window for window in page_of(row.window.page).windows}
    sibling = windows[Path(row.reference).stem]
    return windows[row.id].lines, sibling.lines, page_set / row.reference


def test_region_crops_give_each_region_a_crop_of_another_region_of_its_page_never_its_own_lines(
    page_set: Path, page_cache: Path, built: Path
) -> None:
    entries = {entry.file: entry for entry in read_entries(page_cache)}

    rows = read_region_crops(built / REGION_CROPS)

    assert {entries[row.entry].pair.count("_w") for row in rows} == {0, 1}  # whole pages and windows both served
    for row in rows:
        target, source, _ = _target_and_source(page_set, entries[row.entry].pair)
        targets, sources = _regions(target), _regions(source)
        assert [crop.target for crop in row.crops] == list(targets)  # one a region, in order
        for crop in row.crops:
            assert sources[crop.source].isdisjoint(targets[crop.target])  # never a line of its own region
            if "_w" in row.pair:  # a window's from its sibling, which holds none of its lines at all
                assert sources[crop.source].isdisjoint(_texts(target))
            with Image.open(built / crop.file) as image:
                assert image.size == crop.size


def test_region_crops_leave_out_entries_drawn_without_a_sample_or_in_a_changed_hand_counting_them(
    page_cache: Path, built: Path
) -> None:
    entries = read_entries(page_cache)

    record = RegionCropBuild.read(built / "region_crops.json")

    unsampled = [entry for entry in entries if not entry.variant.takes_reference]
    handed = [entry for entry in entries if entry.variant.takes_reference and entry.hand is not None]
    assert unsampled and handed
    assert (record.entries_read, record.served, record.skipped) == (
        len(entries),
        len(entries) - len(unsampled) - len(handed),
        {Skipped.NO_SAMPLE.value: len(unsampled), Skipped.HAND_CHANGE.value: len(handed)},
    )
    assert {row.entry for row in read_region_crops(built / REGION_CROPS)} == {
        entry.file for entry in entries if entry not in unsampled and entry not in handed
    }


def test_region_crops_take_the_look_change_their_entry_took(page_cache: Path, built: Path) -> None:
    entries = {entry.file: entry for entry in read_entries(page_cache)}
    by_pair: dict[str, list[CacheEntry]] = {}
    for row in read_region_crops(built / REGION_CROPS):
        by_pair.setdefault(row.pair, []).append(entries[row.entry])
    restyled, scanned = next(
        sorted(served, key=lambda entry: entry.restyle is None)
        for served in by_pair.values()
        if len(served) == 2 and sum(entry.restyle is not None for entry in served) == 1
    )
    rows = {row.entry: row for row in read_region_crops(built / REGION_CROPS)}

    pixels = [
        np.asarray(Image.open(built / rows[entry.file].crops[0].file), dtype=float) for entry in (restyled, scanned)
    ]

    assert [crop.source for crop in rows[restyled.file].crops] == [crop.source for crop in rows[scanned.file].crops]
    assert np.abs(pixels[0] - pixels[1]).mean() > 8  # the same crop, one in its entry's look change
