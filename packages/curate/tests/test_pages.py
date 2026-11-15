"""Scans cut into the rows of the single-page table, and a page cropped from its row."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import lance
import pytest
from PIL import Image

from diffu_curate.gutter import Gutter
from diffu_curate.pages import PageRow, open_page, pool_groups, truth_scans, write_pages
from diffu_curate.settings import Pool
from diffu_page.page.layout.pages import digest

OPENING = (2000, 1000)
LEFT = [("vänster sida", (100, 100, 700, 60)), ("andra raden", (100, 200, 700, 60))]
RIGHT = [("höger sida", (1200, 100, 700, 60))]


def rows(table: Path) -> dict[str, PageRow]:
    return {row["page"]: PageRow.model_validate(row) for row in lance.dataset(str(table)).to_table().to_pylist()}


def test_write_pages_cuts_each_scan_into_its_pages(
    pool: Pool, add_scan: Callable[[str, str], Path], alto: Callable[..., str], tmp_path: Path
) -> None:
    add_scan("V1_0001", alto("htrflow", LEFT + RIGHT, size=OPENING))
    add_scan("V1_0002", alto("htrflow", LEFT, size=OPENING))
    add_scan("V1_0003", alto("htrflow", [], size=OPENING))
    add_scan("V1_0004", alto("htrflow", [("a", (50, 100, 300, 60)), ("b", (600, 100, 300, 60))], size=(1000, 1000)))
    add_scan("V1_0005", "<alto><broken")
    table = tmp_path / "pages.lance"

    list(write_pages(pool_groups(pool, table), table, workers=1))

    written = rows(table)
    assert sorted(written) == [
        "V1_0001_left", "V1_0001_right", "V1_0002_left", "V1_0003_whole", "V1_0004_whole", "V1_0005_whole",
    ]  # fmt: skip
    left, right = written["V1_0001_left"], written["V1_0001_right"]
    assert (left.x0, left.x1, left.lines, left.chars, left.alone) == (0, 1000, 2, 23, False)
    assert (right.x0, right.x1, right.y1, right.lines, right.alone) == (1000, 2000, 1000, 1, False)
    # the facing page has no line: the cut is the middle of the band without text, beyond the fold at 1000
    assert (written["V1_0002_left"].x1, written["V1_0002_left"].alone) == (1100, True)
    assert (written["V1_0003_whole"].lines, written["V1_0003_whole"].x1) == (0, 2000)
    # two pages side by side on a scan no wider than tall: no gutter is looked for
    assert (written["V1_0004_whole"].lines, written["V1_0004_whole"].x1) == (2, 1000)
    assert written["V1_0005_whole"].x1 is None
    assert "V1_0005.xml" in (written["V1_0005_whole"].error or "")


def test_write_pages_cuts_a_scan_at_its_gutter_when_given_one(
    pool: Pool, add_scan: Callable[[str, str], Path], alto: Callable[..., str], tmp_path: Path
) -> None:
    add_scan("V1_0001", alto("htrflow", LEFT, size=OPENING))  # the facing page without a line
    add_scan("V1_0002", alto("htrflow", LEFT + RIGHT, size=OPENING))  # the network sees no gutter: one page
    gutters = {
        str(pool.scan_file("V1_0001")): Gutter(score=0.9, x=0.5),
        str(pool.scan_file("V1_0002")): Gutter(score=0.1, x=0.4),
    }
    table = tmp_path / "pages.lance"

    list(write_pages(pool_groups(pool, table, gutters), table, workers=1))

    written = rows(table)
    assert sorted(written) == ["V1_0001_left", "V1_0001_right", "V1_0002_whole"]
    left, right = written["V1_0001_left"], written["V1_0001_right"]
    assert (left.x0, left.x1, left.lines, left.cut, left.alone) == (0, 1000, 2, "image", None)
    assert (right.x0, right.x1, right.lines, right.gutter_score) == (1000, 2000, 0, pytest.approx(0.9))
    assert (written["V1_0002_whole"].x1, written["V1_0002_whole"].lines, written["V1_0002_whole"].cut) == (
        2000,
        3,
        "image",
    )


def test_truth_scans_take_the_ground_truths_ids_and_splits(alto: Callable[..., str], tmp_path: Path) -> None:
    folder = tmp_path / "gt" / "GT-RA_test" / "Volym 1" / "Volym 1"
    (folder / "alto").mkdir(parents=True)
    for stem in ("A1_0001", "A1_0002"):
        (folder / "alto" / f"{stem}.xml").write_text(alto("Transkribus", LEFT + RIGHT, size=OPENING), encoding="utf-8")
        (folder / f"{stem}.jpg").write_bytes(b"jpg")
    (folder / "alto" / "test_0050").write_text("A1_0002.xml\n", encoding="utf-8")
    table = tmp_path / "pages.lance"

    list(write_pages([truth_scans(tmp_path / "gt", "test_0050")], table, workers=1))

    written = rows(table)
    name = digest("GT-RA_test/Volym 1/Volym 1/alto/A1_0001.xml")
    assert set(written) >= {f"{name}_left", f"{name}_right"}
    page = written[f"{name}_left"]
    assert (page.pool, page.volume, page.scan, page.collection, page.split) == (
        "gt", "Volym 1", "A1_0001", "GT-RA_test", "train",
    )  # fmt: skip
    assert sorted(row.split or "" for row in written.values()) == ["test", "test", "train", "train"]


def test_open_page_crops_the_scan_to_the_rows_rectangle(tmp_path: Path) -> None:
    scan = Image.new("RGB", (200, 100), "white")
    scan.paste((255, 0, 0), (100, 0, 200, 100))
    scan.save(tmp_path / "scan.png")
    row = PageRow(
        page="s_right", pool="pool", volume="V", scan="s", side="right", labels="s.xml",
        image=str(tmp_path / "scan.png"), x0=100, y0=0, x1=200, y1=100,
    )  # fmt: skip

    page = open_page(row)

    assert (page.size, page.getpixel((0, 0)), page.getpixel((99, 99))) == ((100, 100), (255, 0, 0), (255, 0, 0))
