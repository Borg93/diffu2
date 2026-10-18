"""Tests for the scan-augmentation config + pipeline.

These guard the two things most likely to break silently: a knob declared but **not wired** into the
pipeline (so retuning does nothing), and a config that accepts a nonsensical value.
"""

from __future__ import annotations

import numpy as np
import pytest
from diffu_page.augment import AugmentConfig
from diffu_page.compose import compose_page_from_inks
from diffu_page.layout import PageSpec, build_layout
from diffu_page.render import PlaceholderRenderer
from pydantic import ValidationError


def _layout_and_inks() -> tuple:
    spec = PageSpec(width=600, height=800)
    layout = build_layout(spec, ["the quick brown fox jumps over the lazy dog"] * 8, seed=1)
    inks = [PlaceholderRenderer().render(b.text, b.h) for b in layout.boxes]
    return layout, inks


@pytest.mark.parametrize(
    "field,value", [("bed_prob", 1.5), ("bed_prob", -0.1), ("clasp_prob", 2.0), ("skew", -1.0)]
)
def test_config_rejects_out_of_range_values(field: str, value: float):
    with pytest.raises(ValidationError):
        AugmentConfig(**{field: value})


def test_config_is_frozen():
    with pytest.raises(ValidationError):
        AugmentConfig().bed_prob = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "low,high",
    [
        (AugmentConfig(bed_prob=0.0), AugmentConfig(bed_prob=1.0)),  # bed mount
        (AugmentConfig(vignette=(0.0, 0.0)), AugmentConfig(vignette=(0.5, 0.5))),  # platen vignette
        (AugmentConfig(skew=0.0), AugmentConfig(skew=0.03)),  # global scanner skew
        (AugmentConfig(dust_prob=0.0), AugmentConfig(dust_prob=1.0)),  # dust + scratches
        (AugmentConfig(stamp_prob=0.0), AugmentConfig(stamp_prob=1.0)),  # archival stamp
        (AugmentConfig(fold_prob=0.0), AugmentConfig(fold_prob=1.0)),  # crease / tide-line
        (AugmentConfig(correction_prob=0.0), AugmentConfig(correction_prob=1.0)),  # strikethroughs / blots
        (AugmentConfig(rotate=(0.0, 0.0)), AugmentConfig(rotate=(3.0, 3.0))),  # whole-page rotation
        (AugmentConfig(microfilm_prob=0.0), AugmentConfig(microfilm_prob=1.0)),  # microfilm modality
        (AugmentConfig(wash=(0.0, 0.0)), AugmentConfig(wash=(0.7, 0.7))),  # low-contrast damp wash
        (AugmentConfig(ink_fade=(0.0, 0.0)), AugmentConfig(ink_fade=(0.9, 0.9))),  # body-ink fading
    ],
)
def test_each_knob_actually_changes_the_page(low: AugmentConfig, high: AugmentConfig):
    """A knob that is declared but not threaded into the pipeline would produce identical pages here."""
    layout, inks = _layout_and_inks()
    a, _ = compose_page_from_inks(layout, inks, seed=4, aug=low)
    b, _ = compose_page_from_inks(layout, inks, seed=4, aug=high)
    assert not np.array_equal(a, b), "changing this knob did not change the output — is it wired?"


def test_microfilm_mode_is_monochrome_and_keeps_labels_on_page():
    layout, inks = _layout_and_inks()
    page, lines = compose_page_from_inks(
        layout, inks, seed=7, aug=AugmentConfig(microfilm_prob=1.0, bed_prob=0.0)
    )
    b, g, r = (page[..., i].astype(np.int16) for i in range(3))
    assert np.abs(b - g).mean() < 3 and np.abs(g - r).mean() < 3, "microfilm should be (near) grayscale"
    assert lines, "labels lost under microfilm"
    for ln in lines:
        for x, y in ln.polygon:
            assert 0 <= x < layout.spec.width and 0 <= y < layout.spec.height


def test_ink_colour_changes_pixels_but_never_the_labels():
    layout, inks = _layout_and_inks()
    black = AugmentConfig(ink_colors=[(0.05, 0.05, 0.06)], ink_fade=(0.0, 0.0), microfilm_prob=0.0)
    brown = AugmentConfig(ink_colors=[(0.06, 0.10, 0.40)], ink_fade=(0.0, 0.0), microfilm_prob=0.0)
    page_a, lines_a = compose_page_from_inks(layout, inks, seed=3, aug=black)
    page_b, lines_b = compose_page_from_inks(layout, inks, seed=3, aug=brown)
    assert not np.array_equal(page_a, page_b), "ink colour did not change the page — is it wired?"
    assert [ln.model_dump() for ln in lines_a] == [ln.model_dump() for ln in lines_b], "ink moved labels"


def test_bed_prob_zero_leaves_a_bright_corner_no_dark_bed():
    layout, inks = _layout_and_inks()
    bedded, _ = compose_page_from_inks(layout, inks, seed=2, aug=AugmentConfig(bed_prob=1.0))
    clean, _ = compose_page_from_inks(layout, inks, seed=2, aug=AugmentConfig(bed_prob=0.0))
    corner = lambda p: float(p[:24, :24].mean())  # noqa: E731 - terse local for a one-off
    assert corner(bedded) < corner(clean) - 20, "bed_prob=1 should darken the frame corner vs bed_prob=0"
