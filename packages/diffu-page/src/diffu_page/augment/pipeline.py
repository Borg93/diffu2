"""The scan pipeline — :func:`scan_pass` runs the ordered stage sequence over a seated page."""

from __future__ import annotations

import random

import numpy as np

from ..labels import LineLabel
from ..layout import PageSpec
from .bed import PaperBounds, _mount_on_bed
from .binding import _binding_shadow, _gutter_focus, _spine_warp, center_shadow
from .config import AugmentConfig
from .geometry import _clamp_lines, _page_rotate, _scanner_warp
from .marks import (
    _curled_corner,
    _dust,
    _gloved_fingers,
    _holes,
    _ink_corrections,
    _margin_marks,
    _paraphs,
    _stamps,
)
from .photometric import _degrade, _folds, _jpeg_recompress, _microfilm


def scan_pass(
    page: np.ndarray,
    lines: list[LineLabel],
    *,
    spec: PageSpec,
    rng: random.Random,
    np_rng: np.random.Generator,
    aug: AugmentConfig,
    trace: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[LineLabel]]:
    """Turn the seated page into one coherent scan — a short, ordered stage pipeline. Add or reorder a
    stage by editing this sequence; the photometric stages leave labels untouched, the geometric ones
    carry the polygons with them, and the clamp guarantees no label leaves the sheet or sits on the binding.

    ``trace`` (optional) is a dict each stage records into (feature tag -> strength) so an audit can verify
    which scan features actually fired on a page; passing ``None`` leaves behaviour byte-identical.
    """
    if spec.spread:  # a two-page opening: shadow the central gutter instead of a single-edge spine
        center_shadow(page, rng, aug)
    elif spec.binding:  # spine shadow over the empty gutter, before any aging
        _binding_shadow(page, spec, rng, aug)
        if trace is not None:
            trace["gutter"] = 1.0  # single-page inner-gutter shadow/curvature (C7)
    _ink_corrections(page, lines, rng, aug)  # strikethroughs / blots ride the warps with the text
    _stamps(page, rng, aug)  # archival stamp / seal
    _margin_marks(page, lines, rng, aug, trace)  # keyed reference marks in the outer margin
    _paraphs(page, lines, rng, aug, trace)  # a scribal flourish at the text-block foot
    grime = rng.random()  # 0 = pristine, 1 = filthy
    out = _degrade(page, grime, rng, np_rng, aug, trace)
    out = _folds(out, rng, aug, trace)  # crease shadow / brown water tide-line
    out, lines = _scanner_warp(out, lines, rng, aug)  # tiny global perspective wobble
    out, lines = _page_rotate(out, lines, rng, aug)  # a few degrees off-square on the bed
    if spec.binding:
        out, lines = _spine_warp(out, lines, rng, spec, aug)  # localized distortion at the binding only
        out = _gutter_focus(out, spec, rng, aug)  # soft focus toward the spine
    lines = _clamp_lines(lines, spec, out.shape[1], out.shape[0])  # never off-page or on the binding
    bounds: PaperBounds | None = None  # set once the sheet is inset on a bed, so holes land on the paper
    if rng.random() < aug.microfilm_prob:  # a filmed scan: bitonal + grain + tramlines (its own frame)
        out = _microfilm(out, rng, np_rng)
        if trace is not None:
            trace["microfilm"] = 1.0
    elif rng.random() < aug.bed_prob:  # else most colour scans show a dark scanner bed; some tightly cropped
        out, lines, bounds = _mount_on_bed(out, lines, rng, np_rng, spec, aug)
        if trace is not None:
            trace["bed"] = 1.0
    _gloved_fingers(out, rng, aug, trace)  # a gloved operator fingertip holding the leaf
    _curled_corner(out, rng, aug, trace)  # a lifted brittle corner
    _holes(out, rng, aug, trace, bounds)  # worm / punch holes in the margin (on the sheet, not the bed)
    _dust(out, rng, aug)  # dust specks + scratches on the final scan
    return _jpeg_recompress(out, grime, rng), lines
