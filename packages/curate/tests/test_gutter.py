"""The gutter network trained on drawn scans, and the gutters of a page table's scans written to their table."""

from __future__ import annotations

import random
from pathlib import Path

import lance
import numpy as np
import pytest
from PIL import Image, ImageDraw

from diffu_curate.gutter import (
    GutterNet,
    find_gutters,
    load_gutter_net,
    save_gutter_net,
    scan_thumbnail,
    train_gutter_net,
    write_gutters,
)
from diffu_curate.pages import PageRow
from diffu_curate.table import append_rows


def draw_scan(path: Path, fold: float | None, seed: int) -> Path:
    """A scan drawn on grey paper: an opening with a dark fold at ``fold`` of its width and lines of "writing" on both
    pages, or (``fold`` None) a single tall sheet of writing."""
    draw_from = random.Random(seed)
    size = (900, 600) if fold is not None else (500, 700)
    scan = Image.new("L", size, draw_from.randint(190, 230))
    pen = ImageDraw.Draw(scan)
    for y in range(60, size[1] - 40, 34):
        for left, right in [(0.06, 0.94)] if fold is None else [(0.05, fold - 0.04), (fold + 0.04, 0.95)]:
            pen.line([(left * size[0], y), (right * size[0] - draw_from.randint(0, 60), y)], fill=40, width=5)
    if fold is not None:
        pen.rectangle([fold * size[0] - 6, 0, fold * size[0] + 6, size[1]], fill=70)
    scan.save(path)
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> GutterNet:
    """A network trained on 64 drawn openings, their folds between 0.35 and 0.65, and 32 drawn sheets."""
    folder = tmp_path_factory.mktemp("train")
    draw_from = random.Random(0)
    folds: list[float | None] = [draw_from.uniform(0.35, 0.65) for _ in range(64)] + [None] * 32
    thumbnails = [scan_thumbnail(str(draw_scan(folder / f"{i}.png", fold, i))) for i, fold in enumerate(folds)]
    return train_gutter_net(np.stack([t for t in thumbnails if t is not None]), folds, epochs=12, batch=16)


def test_a_trained_network_finds_an_openings_fold_and_none_on_a_sheet(trained: GutterNet, tmp_path: Path) -> None:
    opening = scan_thumbnail(str(draw_scan(tmp_path / "opening.png", 0.42, 1000)))
    sheet = scan_thumbnail(str(draw_scan(tmp_path / "sheet.png", None, 1001)))
    assert opening is not None and sheet is not None

    of_opening, of_sheet = find_gutters(trained, np.stack([opening, sheet]))

    assert of_opening.found
    assert of_opening.x == pytest.approx(0.42, abs=0.03)
    assert not of_sheet.found


def test_write_gutters_writes_a_row_for_every_scan_of_the_page_table_once(trained: GutterNet, tmp_path: Path) -> None:
    opening = draw_scan(tmp_path / "V1_0001.png", 0.5, 2000)
    pages = tmp_path / "pages.lance"
    rows = [
        PageRow(
            page=f"V1_0001_{side}", pool="pool", volume="V1", scan="V1_0001", side=side, labels="x", image=str(opening)
        )
        for side in ("left", "right")
    ] + [
        PageRow(
            page="V1_0002_whole", pool="pool", volume="V1", scan="V1_0002", side="whole", labels="x", image="gone.png"
        )
    ]
    append_rows(pages, rows)
    weights = tmp_path / "weights" / "gutter.pt"
    save_gutter_net(trained, weights)
    table = tmp_path / "gutters.lance"

    list(write_gutters(load_gutter_net(weights), pages, table, workers=1))
    list(write_gutters(load_gutter_net(weights), pages, table, workers=1))

    written = {row["scan"]: row for row in lance.dataset(str(table)).to_table().to_pylist()}
    assert lance.dataset(str(table)).count_rows() == 2
    assert written["V1_0001"]["gutter_x"] == pytest.approx(0.5, abs=0.03)
    assert written["V1_0001"]["gutter_score"] >= 0.5
    assert (written["V1_0002"]["gutter_score"], written["V1_0002"]["gutter_x"]) == (None, None)
