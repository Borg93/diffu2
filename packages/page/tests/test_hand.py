"""Measuring a hand from its line boxes: strokes drawn at a known slant and pen width measure as drawn, the measures
read as the caption's words, and two hands compare measure by measure."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from diffu_page.captions.hand import LINE_HEIGHT, HandDistance, HandLook, hand_distance, hand_look, hand_words

PAPER, INK = (232, 220, 192), (58, 42, 28)  # "#E8DCC0", "#3A2A1C"
LINE_BOXES = [[100, 50, 400, 950], [550, 50, 850, 950]]  # two 120 x 540 px lines of a 600 x 400 px page


def _handwriting(slant: float) -> Image.Image:
    """``LINE_BOXES`` filled with 16 strokes a line leaning ``slant`` degrees, 9 px wide: 54 px tall in the core band,
    every fourth one 78 px (an ascender). No anti-aliasing, so the page has exactly two colours."""
    page = np.empty((400, 600, 3), np.uint8)
    page[:] = PAPER
    tan = math.tan(math.radians(slant))
    for top in (40, 220):
        baseline = top + 90
        for k, x in enumerate(range(60, 540, 30)):
            for row in range(baseline - (78 if k % 4 == 0 else 54), baseline):
                left = round(x + (baseline - row) * tan)
                page[row, left : left + 9] = INK
    return Image.fromarray(page)


@pytest.mark.parametrize("slant", [pytest.param(30, id="right"), pytest.param(-10, id="left")])
def test_hand_look_measures_the_strokes_drawn_in_the_line_boxes(slant: int) -> None:
    page = _handwriting(slant)

    hand = hand_look(page, LINE_BOXES)

    assert hand.slant == pytest.approx(slant, abs=1)
    assert hand.stroke == pytest.approx(9 / 120, abs=1 / LINE_HEIGHT)
    assert hand.letters == pytest.approx(54 / 120, abs=1 / LINE_HEIGHT)
    assert hand.density == pytest.approx(9 * (4 * 78 + 12 * 54) / (540 * 120), rel=0.05)
    assert (hand.paper, hand.ink) == ("#E8DCC0", "#3A2A1C")


def test_hand_look_of_line_boxes_without_ink_has_no_slant() -> None:
    page = Image.new("RGB", (600, 400), PAPER)

    hand = hand_look(page, LINE_BOXES)

    assert hand == HandLook(slant=None, stroke=0.0, letters=0.0, density=0.0, paper="#E8DCC0", ink="#000000")


@pytest.mark.parametrize(
    ("slant", "words"),
    [
        pytest.param(
            33.0,
            "right-slanted cursive, about 35 degrees, with fine pen strokes and medium-sized, closely packed letters",
            id="measured",
        ),
        pytest.param(None, "cursive with fine pen strokes and medium-sized, closely packed letters", id="unmeasured"),
    ],
)
def test_hand_words_name_the_slant_pen_letters_and_spacing(slant: float | None, words: str) -> None:
    hand = HandLook(slant=slant, stroke=0.05, letters=0.3, density=0.2, paper="#E8DCC0", ink="#3A2A1C")

    assert hand_words(hand) == words


def test_hand_distance_compares_the_slant_the_lengths_and_the_colours() -> None:
    drawn = HandLook(slant=30.0, stroke=0.06, letters=0.4, density=0.12, paper="#FFFFFF", ink="#3A2A1C")
    reference = HandLook(slant=20.0, stroke=0.12, letters=0.1, density=0.12, paper="#000000", ink="#3A2A1C")

    distance = hand_distance(drawn, reference)

    assert distance.model_dump() == pytest.approx(
        {"slant": 10.0, "stroke": 1.0, "letters": 2.0, "density": 0.0, "paper": 100.0, "ink": 0.0}, abs=0.01
    )


def test_hand_distance_leaves_out_what_one_hand_lacks() -> None:
    blank = HandLook(slant=None, stroke=0.0, letters=0.0, density=0.0, paper="#E8DCC0", ink="#000000")
    reference = HandLook(slant=20.0, stroke=0.08, letters=0.4, density=0.16, paper="#E8DCC0", ink="#000000")

    distance = hand_distance(blank, reference)

    assert distance == HandDistance(slant=None, stroke=None, letters=None, density=None, paper=0.0, ink=0.0)
