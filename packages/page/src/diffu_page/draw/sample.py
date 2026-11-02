"""A sample page sized as a model was trained to read it (its model index's ``sample_size``:
``diffu_moss.train.pairs.SampleSize``), in drawing (``ReferenceGridStep``) and in the training pairs
(``diffu_page.data.page_pair_samples``) alike:

- ``half``: shrunk under half a megapixel on the latent grid (``as_reference``), whatever the target: a 1 MP page's
  sample at half its pixels, a window's letters several times smaller than the window's own;
- ``legible``: at the target's writing scale. A whole page's sample (the target a plan of ``plan_whole_page``) is the
  sample page at the target's pixel count (:func:`sample_at_pixels`); a window's is a window of the sample page at the
  target's line pitch and size (:func:`sample_window`): the sample page scaled so its line pitch is the target's, cut
  around the middle of its lines. A line's box is its slot of one line pitch (``Crop.slot``), so a page's pitch is its
  lines' boxes' median height (:func:`line_pitch`): the target's measured on its plan's lines, the sample's on the line
  boxes it comes with (a page-pair set's whole-page plan of it, as ``val_eval`` gives a sample's lines; a job's
  ``reference.lines``).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from PIL import Image

from diffu_moss.train.pairs import SampleSize
from diffu_page.draw.plan import CropPlan, whole_page_size
from diffu_page.page.crops.geometry import LayoutBox
from diffu_page.page.crops.grid import LATENT_GRID, as_reference


def line_pitch(lines: Sequence[LayoutBox], height: int) -> float:
    """The line pitch, px, of an image ``height`` px tall whose lines' boxes are ``lines`` (0-1000 of it, y first):
    their median height, as a line's box is its slot of one pitch.

    Raises:
        ValueError: If there is no line to measure.
    """
    if not lines:
        raise ValueError("no line box to measure a line pitch by")
    return statistics.median(box[2] - box[0] for box in lines) * height / 1000


def sample_at_pixels(sample: Image.Image, pixels: int) -> Image.Image:
    """``sample`` at about ``pixels`` pixels, its aspect kept, each side on the latent grid, as a page is planned whole
    (``whole_page_size``)."""
    size = whole_page_size(sample.width, sample.height, pixels)
    return sample.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _floored(side: float) -> int:
    """A side floored onto the latent grid (at least one cell)."""
    return max(LATENT_GRID, int(side // LATENT_GRID) * LATENT_GRID)


def sample_window(
    sample: Image.Image, lines: Sequence[LayoutBox], *, pitch: float, size: tuple[int, int]
) -> Image.Image:
    """A window of ``sample``, a page whose lines' boxes are ``lines``, at the line pitch ``pitch`` px and the size
    ``size`` (width, height): the page scaled so that its pitch (:func:`line_pitch`) is ``pitch``, cut at ``size``
    around the middle of its lines' bounding box and inside the page. A side the scaled page is too short for is the
    page's whole side, floored onto the latent grid.

    Raises:
        ValueError: If there is no line box to measure the sample's pitch by.
    """
    scale = pitch / line_pitch(lines, sample.height)  # the window's px per px of the sample
    width, height = (min(side, _floored(extent * scale)) for side, extent in zip(size, sample.size, strict=True))
    across, down = width / scale, height / scale  # the window on the sample, px
    middle_x = (min(box[1] for box in lines) + max(box[3] for box in lines)) / 2000 * sample.width
    middle_y = (min(box[0] for box in lines) + max(box[2] for box in lines)) / 2000 * sample.height
    left = min(max(0.0, middle_x - across / 2), sample.width - across)
    top = min(max(0.0, middle_y - down / 2), sample.height - down)
    window = (left, top, left + across, top + down)
    return sample.convert("RGB").resize((width, height), Image.Resampling.LANCZOS, box=window)


def legible_sample(
    sample: Image.Image, lines: Sequence[LayoutBox], *, size: tuple[int, int], pitch: float | None
) -> Image.Image:
    """``sample``, a page whose lines' boxes are ``lines``, at the writing scale of a target of ``size`` (width,
    height) px: for a whole page (``pitch`` None) at the target's pixel count (:func:`sample_at_pixels`), for a window
    at its line ``pitch`` px and its size (:func:`sample_window`).

    Raises:
        ValueError: If the target is a window and there is no line box to measure the sample's pitch by.
    """
    if pitch is None:
        return sample_at_pixels(sample, size[0] * size[1])
    return sample_window(sample, lines, pitch=pitch, size=size)


def sized_sample(sample: Image.Image, rule: SampleSize, *, plan: CropPlan, lines: Sequence[LayoutBox]) -> Image.Image:
    """``sample``, a page whose lines' boxes are ``lines`` (none if not given), sized by ``rule`` for drawing ``plan``:
    shrunk (``as_reference``), or at the plan's writing scale (:func:`legible_sample`; a window's pitch measured on
    its own lines).

    Raises:
        ValueError: If the rule is legible, the plan a window, and the sample comes without its line boxes.
    """
    if rule is SampleSize.HALF:
        return as_reference(sample)
    if plan.whole_page:
        return legible_sample(sample, lines, size=(plan.width, plan.height), pitch=None)
    if not lines:
        raise ValueError(
            "this model reads a window's sample cut at the window's line pitch from the sample page's lines: give the "
            "sample's line boxes (a page job's reference.lines), or draw the page whole"
        )
    pitch = line_pitch([line.bbox for line in plan.lines], plan.height)
    return legible_sample(sample, lines, size=(plan.width, plan.height), pitch=pitch)
