"""Invariant tests for the augmentation conditions in ``AUGMENTATION_SPEC.md``.

These lock the properties whose violation silently corrupts an HTR training set or breaks a corpus-match
condition: ink that fills its slot (but never overruns it or blows up a ragged line), physical-handling
marks that never move a label off the page, a feature trace that honestly reports what fired, scan profiles
that compose valid pages across a real tone range, and the sparse letter layout. All run on the CPU
:class:`PlaceholderRenderer` — no model, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from diffu_page.augment import AugmentConfig
from diffu_page.compose import compose_page
from diffu_page.layout import PageSpec, build_layout
from diffu_page.profiles import PROFILE_NAMES, profile
from diffu_page.render import PlaceholderRenderer

_SENT = "Goteborgs poliskammare den fjarde maj artonhundra attiotva anhallen for fylleri"
_FLAT = AugmentConfig(  # every geometric/photometric mark off -> ink fill is measured cleanly
    bed_prob=0.0,
    skew=0.0,
    rotate=(0.0, 0.0),
    microfilm_prob=0.0,
    dust_prob=0.0,
    correction_prob=0.0,
    stamp_prob=0.0,
    fold_prob=0.0,
    glove_prob=0.0,
    hole_prob=0.0,
    curl_prob=0.0,
)


def _body_fills(seed: int = 0) -> tuple[list[float], float]:
    """Compose a full paragraph + one deliberately short line; return (per-body-line slot fill, max_stretch)."""
    spec = PageSpec(width=1400, height=1900, template="single_column", line_height=40, line_gap=0.4)
    text = [" ".join([_SENT] * 8), "kort"]
    renderer = PlaceholderRenderer()
    layout = build_layout(spec, text, seed=seed, measure=renderer.measure_width)
    body_w = max((b.w for b in layout.boxes if b.region == "body"), default=1)
    _, lines = compose_page(layout, renderer, seed=seed, aug=_FLAT)
    fills = [
        (max(p[0] for p in ln.polygon) - min(p[0] for p in ln.polygon)) / body_w
        for ln in lines
        if ln.region == "body"
    ]
    return fills, spec.max_stretch


def test_full_body_lines_fill_their_slot_but_ragged_lines_stay_ragged():
    fills, _ = _body_fills()
    assert len(fills) >= 4, "expected several body lines"
    full = sorted(fills, reverse=True)
    assert full[2] >= 0.90, f"the full body lines must fill >=0.90 of the slot; got {full[:3]}"
    assert min(fills) < 0.60, "a short/ragged line must stay ragged, not be blown up to fill the slot"


def test_no_body_line_overruns_its_slot():
    fills, _ = _body_fills()
    assert max(fills) <= 1.01, f"a body line spilled past its slot width: {max(fills):.3f}"


@pytest.mark.parametrize("seed", [0, 1, 5, 9])
def test_forced_physical_features_never_move_a_label_off_the_page(seed: int):
    spec = PageSpec(width=900, height=1240, template="single_column", line_height=40, binding=True)
    aug = AugmentConfig(glove_prob=1.0, hole_prob=1.0, curl_prob=1.0, fold_prob=1.0)
    layout = build_layout(
        spec, [" ".join([_SENT] * 5)] * 3, seed=seed, measure=PlaceholderRenderer().measure_width
    )
    _, lines = compose_page(layout, PlaceholderRenderer(), seed=seed, aug=aug)
    for ln in lines:
        for x, y in [*ln.polygon, *(p for w in ln.words for p in w.polygon)]:
            assert 0 <= x < spec.width and 0 <= y < spec.height, f"feature pushed a label to ({x},{y})"


def test_trace_reports_the_features_that_fired():
    spec = PageSpec(width=900, height=1240, template="single_column", line_height=40, binding=True)
    aug = AugmentConfig(glove_prob=1.0, hole_prob=1.0, curl_prob=1.0, fold_prob=1.0)
    trace: dict[str, float] = {}
    layout = build_layout(
        spec, [" ".join([_SENT] * 5)] * 3, seed=2, measure=PlaceholderRenderer().measure_width
    )
    compose_page(layout, PlaceholderRenderer(), seed=2, aug=aug, trace=trace)
    assert trace.get("gloved_finger") == 1.0, "a forced gloved finger must be traced"
    assert trace.get("hole") == 1.0, "a forced hole must be traced"
    assert trace.get("curled_corner") == 1.0, "a forced curled corner must be traced"
    assert "fold" in trace or "water_tide" in trace, "a forced fold must trace a crease or a tide-line"
    assert trace.get("gutter") == 1.0, "a bound single page must trace its inner gutter"


def test_trace_stays_empty_when_nothing_is_forced_and_is_optional():
    # a fully pristine page fires none of the physical marks; and omitting trace is a no-op (never raises)
    spec = PageSpec(width=800, height=1100)
    layout = build_layout(spec, [_SENT], seed=0)
    trace: dict[str, float] = {}
    compose_page(layout, PlaceholderRenderer(), seed=0, aug=_FLAT, trace=trace)
    assert not {"gloved_finger", "hole", "curled_corner", "fold"} & set(trace), (
        "no marks should fire when off"
    )
    compose_page(layout, PlaceholderRenderer(), seed=0, aug=_FLAT)  # trace=None path must not raise


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_every_scan_profile_composes_a_valid_reproducible_page(name: str):
    spec = PageSpec(width=1000, height=1400, template="single_column", line_height=34)
    prof = profile(name)
    layout = build_layout(
        spec, [" ".join([_SENT] * 6)] * 4, seed=3, measure=PlaceholderRenderer().measure_width
    )
    page_a, lines_a = compose_page(
        layout, PlaceholderRenderer(), seed=3, aug=prof.aug, paper_tone=prof.paper_tone
    )
    page_b, lines_b = compose_page(
        layout, PlaceholderRenderer(), seed=3, aug=prof.aug, paper_tone=prof.paper_tone
    )
    assert lines_a, f"profile {name!r} produced no labels"
    assert np.array_equal(page_a, page_b), f"profile {name!r} is not reproducible"
    assert [ln.model_dump() for ln in lines_a] == [ln.model_dump() for ln in lines_b]
    for ln in lines_a:
        for x, y in ln.polygon:
            assert 0 <= x < spec.width and 0 <= y < spec.height, f"{name!r} label left the page"


def test_profiles_span_a_wide_paper_luminance_range():
    # bitonal (bright near-white) vs faded_gray (dim, washed) must land far apart -> C2's tone spread
    spec = PageSpec(width=900, height=1240, template="single_column", line_height=40)
    layout = build_layout(
        spec, [" ".join([_SENT] * 5)] * 3, seed=1, measure=PlaceholderRenderer().measure_width
    )

    def central_luma(name: str) -> float:
        prof = profile(name)
        page, _ = compose_page(
            layout, PlaceholderRenderer(), seed=1, aug=prof.aug, paper_tone=prof.paper_tone
        )
        h, w = page.shape[:2]
        crop = page[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]
        return float(crop.mean())

    assert central_luma("bitonal") - central_luma("faded_gray") > 30.0, "profiles must span a wide tone range"


def test_letter_layout_is_sparse_with_a_signature():
    spec = PageSpec(width=1200, height=1700, template="letter", line_height=46)
    layout = build_layout(
        spec, [" ".join([_SENT] * 4)] * 2, seed=0, measure=PlaceholderRenderer().measure_width
    )
    _, lines = compose_page(layout, PlaceholderRenderer(), seed=0, aug=AugmentConfig(bed_prob=0.0))
    assert any(ln.region == "signature" for ln in lines), "a letter must carry a signature line"
    dense = build_layout(
        PageSpec(width=1200, height=1700, template="single_column", line_height=24),
        [" ".join([_SENT] * 6)] * 8,
        seed=0,
        measure=PlaceholderRenderer().measure_width,
    )
    _, dense_lines = compose_page(dense, PlaceholderRenderer(), seed=0, aug=AugmentConfig(bed_prob=0.0))
    assert len(lines) < len(dense_lines), "the letter layout must be sparser than a dense column"


def test_a_single_page_is_a_portrait_leaf():
    spec = PageSpec()
    assert spec.height > spec.width and 1.1 <= spec.height / spec.width <= 2.2, (
        "default page must be portrait"
    )
