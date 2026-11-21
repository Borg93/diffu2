"""Line samples: each line box of a target gets a crop of another line of its page, cut at the target's line pitch and
placed on the DiT's position grid over its box or beside the page in its rows."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from diffu_moss.transformer import token_positions
from diffu_page.captions.prompt import LineRecord
from diffu_page.draw.line_samples import CropPlace, cut_line_crops, plan_line_crops, plan_region_crops

SOURCE = (600, 1000)  # px: a page of five lines, each a bar of its own colour, at a 60 px pitch
TARGET = (320, 400)  # px: a window of its first three lines at a 48 px pitch, 25 latent rows and 20 columns
PAPER = (225, 210, 180)


def _ink(k: int) -> tuple[int, int, int]:
    return (30 + 20 * k, 20 + 10 * k, 15)


def _source_page() -> tuple[Image.Image, list[LineRecord]]:
    """The source page and its lines: line ``k`` a bar 30 px tall above its baseline at 130 + 60k px, x 80-520, its
    box its slot of one pitch (0.7 above its baseline, 0.3 below; 0-1000 of the page, y first)."""
    page = Image.new("RGB", SOURCE, PAPER)
    draw = ImageDraw.Draw(page)
    lines = []
    for k in range(5):
        baseline = 130 + 60 * k
        draw.rectangle((80, baseline - 30, 520, baseline - 1), fill=_ink(k))
        box = (baseline - 42, round(1000 * 70 / SOURCE[0]), baseline + 18, 900)
        lines.append(LineRecord(id=f"line_{k}", text=f"line {k} of the page", role="", bbox=box))
    return page, lines


def _target_lines() -> list[LineRecord]:
    """The window's three lines (the page's first three, by their texts), each a 48 px slot from y 16 + 48k px,
    x 32-288."""
    return [
        LineRecord(
            id=f"window_{k}", text=f"line {k} of the page", role="", bbox=(40 + 120 * k, 100, 160 + 120 * k, 900)
        )
        for k in range(3)
    ]


def test_line_crops_cut_the_lines_the_target_does_not_hold_at_its_pitch_one_for_each_of_its_lines() -> None:
    page, lines = _source_page()
    crops = plan_line_crops(SOURCE, lines, TARGET, _target_lines(), seed=3)

    images = cut_line_crops(page, crops)

    assert [crop.target for crop in crops] == ["window_0", "window_1", "window_2"]
    assert {crop.source for crop in crops} == {
        "line_3",
        "line_4",
    }  # the page's other lines, the first again for a third
    assert crops[2].source == crops[0].source
    # a slot of 60 px at the window's 48 px pitch, as wide as its box (256 px) of the line's 376
    assert [(crop.size, crop.tokens) for crop in crops] == [((256, 48), 48)] * 3
    for crop, image in zip(crops, images, strict=True):
        ink = _ink(int(crop.source[-1]))
        column = np.asarray(image)[:, 128]
        bar = [y for y in range(image.height) if column[y].sum() < sum(PAPER) - 60]
        assert image.size == crop.size
        assert len(bar) == pytest.approx(30 * 0.8, abs=1)  # the bar's 30 px at the window's scale
        assert np.abs(column[bar[len(bar) // 2]].astype(int) - ink).max() <= 3  # the source line's own ink


@pytest.mark.parametrize(
    ("place", "top_left"),
    [
        pytest.param("over", (1, 2), id="over-its-box"),  # the first box's top-left cell: 16 px down, 32 px in
        pytest.param("beside", (1, -16), id="beside-the-page-in-its-rows"),  # its 16 columns just left of column 0
    ],
)
def test_line_crop_lies_on_the_position_grid_at_its_box_s_rows(place: CropPlace, top_left: tuple[int, int]) -> None:
    _, lines = _source_page()
    rows, columns = TARGET[1] // 16, TARGET[0] // 16
    crop = plan_line_crops(SOURCE, lines, TARGET, _target_lines(), place=place)[0]

    positions = token_positions(rows, columns, 1, torch.device("cpu"), references=[(3, 16)], offsets=[crop.offset])

    first = positions[rows * columns + 1]  # the crop's first token, after the drawn image's and the prompt's one
    expected = [top_left[0] - (rows - 1) / 2, top_left[1] - (columns - 1) / 2]  # the drawn grid's cell there
    assert first[2:].tolist() == expected


def test_line_crops_are_none_when_the_source_holds_only_the_targets_lines() -> None:
    _, lines = _source_page()

    crops = plan_line_crops(SOURCE, lines[:3], TARGET, _target_lines())

    assert crops == []


def _two_region_page() -> list[LineRecord]:
    """A 320 x 640 px page of two text regions, ``a`` and ``b``, of five lines each: line ``k`` a 48 px slot from
    y 32 + 48k px, x 32-288."""
    return [
        LineRecord(
            id=f"line_{k}",
            text=f"line {k} of the page",
            role="",
            bbox=(round((32 + 48 * k) / 640 * 1000), 100, round((80 + 48 * k) / 640 * 1000), 900),
            block="a" if k < 5 else "b",
        )
        for k in range(10)
    ]


@pytest.mark.parametrize(
    ("place", "top_left"),
    [
        pytest.param("over", (2, 2), id="over-its-box"),  # region a's box from y 32 px, x 32 px
        pytest.param("beside", (2, -16), id="beside-the-page-in-its-rows"),  # its 16 columns just left of column 0
    ],
)
def test_region_crops_of_a_page_drawn_from_itself_give_each_region_another_region_on_its_box_s_rows(
    place: CropPlace, top_left: tuple[int, int]
) -> None:
    lines = _two_region_page()
    rows, columns = 640 // 16, 320 // 16

    crops = plan_region_crops((320, 640), lines, (320, 640), lines, place=place)

    positions = token_positions(rows, columns, 1, torch.device("cpu"), references=[(15, 16)], offsets=[crops[0].offset])
    first = positions[rows * columns + 1]
    assert [(crop.target, crop.source) for crop in crops] == [("a", "b"), ("b", "a")]  # never a region's own
    assert [crop.size for crop in crops] == [(256, 240)] * 2  # its box: five 48 px slots, 256 px wide
    assert first[2:].tolist() == [top_left[0] - (rows - 1) / 2, top_left[1] - (columns - 1) / 2]
