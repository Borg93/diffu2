"""The writing layer's comparison of where writing is with where the labels say lines are."""

from __future__ import annotations

import torch

from diffu_curate.alto import AltoLine
from diffu_curate.writing import coverage, fitted_size, label_mask


def line(top: float, bottom: float) -> AltoLine:
    """A labelled line across a 400 x 400 page from ``top`` to ``bottom``."""
    return AltoLine(text="x", box=(40, top, 360, bottom), outline=((40, top), (360, top), (360, bottom), (40, bottom)))


def bands(*rows: tuple[int, int]) -> torch.Tensor:
    """A 100 x 100 writing map with writing across each (top, bottom) band."""
    mask = torch.zeros(1, 100, 100, dtype=torch.bool)
    for top, bottom in rows:
        mask[0, top:bottom, 10:90] = True
    return mask


def labels(*lines: AltoLine) -> torch.Tensor:
    return torch.from_numpy(label_mask(lines, (400, 400), (100, 100)))[None]


def test_coverage_counts_writing_no_label_covers_as_uncovered() -> None:
    # two written bands of the same size, only the upper one labelled: half of the writing is under a label
    found = coverage(bands((10, 20), (60, 70)), labels(line(40, 80)))[0]

    assert abs(found.covered - 0.5) < 0.02
    assert found.filled > 0.85  # a drawn outline includes its edge pixels, so the label is a row and a column larger


def test_coverage_counts_a_label_over_blank_paper_as_unfilled() -> None:
    # one written band, two labelled lines far apart: the lower label lies on blank paper
    found = coverage(bands((10, 20)), labels(line(40, 80), line(240, 280)))[0]

    assert abs(found.filled - 0.45) < 0.03  # half, less the outlines' edge pixels
    assert found.covered > 0.95


def test_coverage_takes_the_gap_between_two_labelled_lines_as_labelled() -> None:
    # writing runs through the 5 map pixels between two labelled lines (a descender, a line the outlines pinch)
    found = coverage(bands((10, 35)), labels(line(40, 80), line(100, 140)))[0]

    assert found.covered > 0.99


def test_fitted_size_keeps_the_longer_side_at_1024_and_both_sides_on_the_patch_grid() -> None:
    assert fitted_size(6000, 4000) == (1024, 688)
    assert fitted_size(3000, 4500) == (688, 1024)
