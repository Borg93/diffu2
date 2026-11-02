"""The reference grid: a reference image is drawn from as training saw it, on the 16 px grid and at most the size the
references were trained at; or, for a model trained on legible samples, at its target's writing scale: a whole page's
sample at the target's pixel count, a window's cut from the sample page at the window's line pitch and size."""

from __future__ import annotations

import statistics

import numpy as np
import pytest
from PIL import Image, ImageDraw

from diffu_page.draw.plan import whole_page_size
from diffu_page.draw.sample import legible_sample
from diffu_page.page.crops.geometry import LayoutBox
from diffu_page.page.crops.grid import REF_PIXELS, as_reference


def test_as_reference_squashes_a_small_reference_onto_the_16_px_grid() -> None:
    reference = Image.linear_gradient("L").resize((50, 34)).convert("RGB")  # off the 16 px grid

    gridded = as_reference(reference)

    assert gridded.size == (48, 32)
    assert gridded.tobytes() == reference.resize((48, 32), Image.Resampling.LANCZOS).tobytes()


def test_as_reference_shrinks_a_full_crop_to_the_size_references_were_trained_at() -> None:
    crop = Image.linear_gradient("L").resize((1472, 640)).convert("RGB")  # a 0.94 MP training-bucket crop

    reference = as_reference(crop)

    assert reference.size == (1088, 464)
    assert reference.width * reference.height <= REF_PIXELS


def test_as_reference_leaves_a_reference_sized_image_unchanged() -> None:
    reference = as_reference(Image.linear_gradient("L").resize((1472, 640)).convert("RGB"))

    again = as_reference(reference)

    assert (again.size, again.tobytes()) == (reference.size, reference.tobytes())


SAMPLE_PAGE = (600, 1000)  # px: a sample page, its lines' ink every 60 px
SAMPLE_PITCH = 60


def _sample_page() -> tuple[Image.Image, list[LayoutBox]]:
    """A sample page of 15 lines, each a dark bar 30 px tall above its baseline at a 60 px pitch from y 130, and the
    lines' boxes as its whole-page plan gives them: each its slot of one pitch, 0.7 above its baseline and 0.3 below
    (0-1000 of the page, y first)."""
    width, height = SAMPLE_PAGE
    page = Image.new("RGB", SAMPLE_PAGE, (225, 210, 180))
    draw = ImageDraw.Draw(page)
    boxes: list[LayoutBox] = []
    for k in range(15):
        baseline = 130 + SAMPLE_PITCH * k
        draw.rectangle((80, baseline - 30, 520, baseline - 1), fill=(50, 35, 25))
        top, bottom = baseline - 0.7 * SAMPLE_PITCH, baseline + 0.3 * SAMPLE_PITCH
        boxes.append((round(1000 * top / height), round(1000 * 70 / width), round(1000 * bottom / height), 900))
    return page, boxes


def _bar_pitch(image: Image.Image) -> float:
    """The median distance, px, between the tops of the dark bars an image's rows show."""
    dark = (np.asarray(image.convert("L")) < 128).mean(axis=1) > 0.3
    tops = [y for y in range(1, len(dark)) if dark[y] and not dark[y - 1]]
    return statistics.median(b - a for a, b in zip(tops, tops[1:], strict=False))


@pytest.mark.parametrize(
    ("size", "cut"),
    [
        pytest.param((192, 240), (192, 240), id="at-the-windows-size"),  # 480 x 600 px of the sample page
        pytest.param((192, 480), (192, 400), id="a-side-the-page-is-too-short-for"),  # its 1000 px at 0.4: 400 px
    ],
)
def test_legible_sample_of_a_window_is_cut_from_the_sample_page_at_the_windows_line_pitch(
    size: tuple[int, int], cut: tuple[int, int]
) -> None:
    page, boxes = _sample_page()

    sample = legible_sample(page, boxes, size=size, pitch=24.0)

    assert sample.size == cut
    assert _bar_pitch(sample) == pytest.approx(24, abs=0.5)  # the page's 60 px pitch at the window's 24


def test_legible_sample_of_a_whole_page_is_the_sample_page_at_the_targets_pixel_count() -> None:
    page, boxes = _sample_page()
    target = (816, 1232)  # a 1 MP page

    sample = legible_sample(page, boxes, size=target, pitch=None)

    # its aspect kept and its sides on the grid, as a page is planned whole: 1.1 % more pixels than the target's
    assert sample.size == whole_page_size(*SAMPLE_PAGE, target[0] * target[1]) == (784, 1296)
    assert _bar_pitch(sample) == pytest.approx(SAMPLE_PITCH * 1296 / 1000, abs=0.5)  # the whole page, scaled
