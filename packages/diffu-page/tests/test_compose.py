"""Tests for seating lines on paper + the ground-truth labels that come out.

The invariants here are the ones whose violation silently corrupts an HTR training set: a label that
drifts off the page, a transcription that gets garbled or reordered, a page that isn't reproducible from
its seed, or a word box that doesn't lie on its line.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from diffu_page.augment import AugmentConfig
from diffu_page.compose import compose_page
from diffu_page.layout import PageSpec, build_layout
from diffu_page.render import PlaceholderRenderer

TEXTS = [
    "Goteborgs poliskammare den fjarde maj artonhundra attiotva",
    "anhallen for fylleri och forargelsevackande beteende",
    "vittnen instamde och saken hanskots till rattens provning",
] * 4


def _compose(seed: int, *, binding: bool = False, recto: bool = True, aug: AugmentConfig | None = None):
    spec = PageSpec(width=640, height=860, binding=binding, recto=recto)
    layout = build_layout(spec, TEXTS, seed=seed)
    page, lines = compose_page(layout, PlaceholderRenderer(), seed=seed, aug=aug)
    return spec, layout, page, lines


def _is_subsequence(sub: list[str], full: list[str]) -> bool:
    i = 0
    for f in full:
        if i < len(sub) and sub[i] == f:
            i += 1
    return i == len(sub)


def test_labels_preserve_text_verbatim_and_per_region_reading_order():
    # global order is now reading order (page-number/marginalia no longer precede body), so the invariant
    # is per-region: no text is garbled/synthesized, and within a region order follows the laid-out flow.
    _, layout, _, lines = _compose(1)
    assert lines, "expected at least one label"
    assert all(ln.text.strip() for ln in lines), "no empty labels"
    label_counts, box_counts = Counter(ln.text for ln in lines), Counter(b.text for b in layout.boxes)
    assert all(label_counts[t] <= box_counts[t] for t in label_counts), "a label text isn't from the layout"
    for region in {ln.region for ln in lines}:
        in_region = sorted((ln for ln in lines if ln.region == region), key=lambda ln: ln.order)
        seq = [ln.text for ln in in_region]
        box_seq = [b.text for b in layout.boxes if b.region == region]
        assert _is_subsequence(seq, box_seq), f"region {region!r} lines garbled/reordered vs the layout"


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("binding", [False, True])
@pytest.mark.parametrize("bed", [0.0, 1.0])
def test_gt_polygons_stay_within_the_page(seed: int, binding: bool, bed: float):
    spec, _, _, lines = _compose(seed, binding=binding, aug=AugmentConfig(bed_prob=bed))
    for ln in lines:
        for x, y in [*ln.polygon, *(p for w in ln.words for p in w.polygon)]:
            assert 0 <= x < spec.width and 0 <= y < spec.height, f"label point ({x},{y}) left the page"


@pytest.mark.parametrize("recto", [True, False])
def test_bound_text_stays_off_the_reserved_gutter(recto: bool):
    # isolate the clamp invariant from the bed's uniform rescale
    spec, _, _, lines = _compose(3, binding=True, recto=recto, aug=AugmentConfig(bed_prob=0.0))
    xs = [x for ln in lines for x, _ in ln.polygon]
    if recto:  # spine on the left -> nothing left of margin + gutter
        assert min(xs) >= spec.margin + spec.binding_gutter
    else:  # spine on the right -> nothing right of the mirror bound
        assert max(xs) <= spec.width - spec.margin - spec.binding_gutter


def test_pages_are_reproducible_from_their_seed():
    _, _, page_a, lines_a = _compose(11, binding=True)
    _, _, page_b, lines_b = _compose(11, binding=True)
    assert np.array_equal(page_a, page_b), "same seed produced different pixels"
    assert [ln.model_dump() for ln in lines_a] == [ln.model_dump() for ln in lines_b], "labels differ"


def test_page_composer_streams_lines_and_matches_the_batch_path():
    # PageComposer (the streaming studio's seam) must seat lines one at a time — the count only ever grows —
    # and seating all + finish() must be byte-identical to compose_page_from_inks (same page, same labels).
    from diffu_page.compose import PageComposer, compose_page_from_inks
    from diffu_page.render import render_boxes

    spec = PageSpec(width=640, height=860)
    layout = build_layout(spec, TEXTS, seed=3)
    inks = render_boxes(layout.boxes, PlaceholderRenderer())

    composer = PageComposer(layout, seed=3, aug=AugmentConfig())
    counts = []
    for box, ink in zip(layout.boxes, inks, strict=True):
        composer.seat(box, ink)
        counts.append(len(composer.lines))
    page_stream, lines_stream = composer.finish()

    assert counts == sorted(counts) and counts[-1] >= 1, "seated-line count must grow monotonically"
    page_batch, lines_batch = compose_page_from_inks(layout, inks, seed=3, aug=AugmentConfig())
    assert np.array_equal(page_stream, page_batch), "streamed page differs from the batch page"
    assert [ln.model_dump() for ln in lines_stream] == [ln.model_dump() for ln in lines_batch], (
        "labels differ"
    )


def test_spread_lays_out_two_leaves_read_left_then_right():
    spec = PageSpec(width=1000, height=720, template="spread")
    cx = spec.width // 2
    found_both = False
    for seed in range(8):  # one leaf is occasionally blank — scan the seeds for a full opening
        layout = build_layout(spec, TEXTS, seed=seed)
        _, lines = compose_page(layout, PlaceholderRenderer(), seed=seed, aug=AugmentConfig(bed_prob=0.0))
        left = [ln for ln in lines if ln.region == "left"]
        right = [ln for ln in lines if ln.region == "right"]
        if not (left and right):
            continue
        found_both = True
        assert max(ln.order for ln in left) < min(ln.order for ln in right), "left leaf must read first"
        assert max(p[0] for ln in left for p in ln.polygon) < cx + 15, "left leaf crossed the gutter"
        assert min(p[0] for ln in right for p in ln.polygon) > cx - 15, "right leaf crossed the gutter"
    assert found_both, "no spread produced both leaves across the seeds"


def test_manuscript_edits_are_emitted_as_tagged_gt():
    spec = PageSpec(width=720, height=1000, insertion_prob=1.0, deletion_prob=1.0)
    layout = build_layout(spec, TEXTS, seed=1)
    _, lines = compose_page(layout, PlaceholderRenderer(), seed=1, aug=AugmentConfig(bed_prob=0.0))
    adds = [ln for ln in lines if ln.tag == "add"]
    dels = [ln for ln in lines if ln.tag == "del"]
    assert adds, "interlinear insertions should be tagged add"
    assert all(ln.region == "insertion" for ln in adds), "an insertion is its own region"
    assert dels, "struck lines should be tagged del"
    assert all(ln.text.strip() for ln in dels), "struck text is still transcribed (HTR reads around it)"


def test_no_edits_by_default():
    layout = build_layout(PageSpec(width=720, height=1000), TEXTS, seed=1)
    _, lines = compose_page(layout, PlaceholderRenderer(), seed=1, aug=AugmentConfig(bed_prob=0.0))
    assert all(ln.tag == "" for ln in lines), "edits must be opt-in (insertion_prob/deletion_prob default 0)"


def test_ink_matte_zeros_the_halo_so_lines_have_no_box_seam():
    # the model paints ink on a tan writing-region halo; if the halo gets any alpha it composites as a
    # rectangular box around the line. The matte must zero the halo (and white), keeping only the ink.
    from diffu_page.render import ink_matte

    gray = np.full((40, 200), 205.0, np.float32)  # tan halo = the model's line background
    gray[:, :8] = gray[:, 192:] = 250.0  # a white margin
    gray[18:22, 40:160] = 30.0  # a dark ink stroke
    a = ink_matte(gray)
    assert a[10, 100] < 0.05, "halo got alpha -> a visible box seams around the line"
    assert a[10, 3] < 0.05, "white margin must be transparent too"
    assert a[20, 100] > 0.5, "the ink stroke must stay opaque"


def test_ink_matte_keeps_the_whole_stroke_not_just_the_dark_core():
    # the bug: a flat threshold kept only the darkest core and ate the lighter flanks, thinning the
    # handwriting into broken fragments. The matte must keep the WHOLE stroke (core + lighter flanks).
    from diffu_page.render import ink_matte

    gray = np.full((48, 220), 215.0, np.float32)  # grey writing-region wash (well below white)
    gray[:, :6] = 250.0  # white margin
    gray[20:28, 30:190] = 140.0  # the lighter stroke flank (mid-grey, darker than the wash)
    gray[23:25, 30:190] = 80.0  # the dark stroke core
    a = ink_matte(gray)
    assert a[10, 100] < 0.05, "the flat wash must stay transparent (no box)"
    assert a[24, 100] > 0.7, "the dark stroke core must be opaque"
    assert a[21, 100] > 0.35, "the LIGHTER stroke flank must be kept too (don't thin the strokes)"


def test_ink_matte_suppresses_a_solid_band_but_keeps_strokes():
    # a solid dark band (scan smudge / model wash) must be dropped; a thin stroke must survive.
    from diffu_page.render import ink_matte

    gray = np.full((64, 300), 245.0, np.float32)  # near-white background
    gray[20:48, 80:200] = 120.0  # a SOLID dark band (28px tall x 120 wide) = a blob, not writing
    gray[10:54, 30:34] = 60.0  # a thin tall stroke elsewhere
    a = ink_matte(gray)
    assert a[34, 140] < 0.1, "the solid band must be suppressed"
    assert a[30, 32] > 0.4, "a thin stroke must survive blob suppression"


def test_center_shadow_darkens_the_middle_gutter():
    import random

    from diffu_page.augment import center_shadow

    page = np.full((50, 200, 3), 240, np.uint8)
    center_shadow(page, random.Random(0), AugmentConfig())
    assert float(page[:, 95:105].mean()) < float(page[:, :20].mean()) - 10, "center gutter not shadowed"


def test_word_boxes_are_ordered_left_to_right_within_their_line():
    _, _, _, lines = _compose(2)
    multiword = [ln for ln in lines if len(ln.text.split()) > 1]
    assert multiword, "expected multi-word lines to exercise the word split"
    for ln in multiword:
        line_xs = [p[0] for p in ln.polygon]  # polygon is now a tight hull, not a 4-pt rect
        lx0, lx1 = min(line_xs), max(line_xs)
        starts = [w.polygon[0][0] for w in ln.words]
        assert starts == sorted(starts), "word boxes not left-to-right"
        assert lx0 - 1 <= min(starts) and max(w.polygon[1][0] for w in ln.words) <= lx1 + 1, "word off line"


def test_every_line_has_a_baseline_inside_its_polygon():
    _, _, _, lines = _compose(5)
    for ln in lines:
        assert len(ln.baseline) >= 2, "missing baseline"
        ys = [p[1] for p in ln.polygon]
        by = ln.baseline[0][1]
        assert min(ys) - 2 <= by <= max(ys) + 2, "baseline not within the line's vertical extent"
        assert ln.baseline[0][0] <= ln.baseline[-1][0], "baseline not left-to-right"


def test_reading_order_is_a_dense_unique_sequence():
    _, _, _, lines = _compose(5)
    orders = sorted(ln.order for ln in lines)
    assert orders == list(range(len(lines))), "reading order must be a 0..n-1 permutation"


def test_page_number_and_header_read_before_body_marginalia_after():
    # a furnished page: explicit reading order must put page-number/header ahead of body, marginalia last
    spec = PageSpec(
        width=720,
        height=1000,
        page_number="7.",
        header_lines=["Goteborgs Poliskammare"],
        marginalia=["N:o 4", "vide"],
    )
    layout = build_layout(spec, TEXTS, seed=4)
    _, lines = compose_page(layout, PlaceholderRenderer(), seed=4, aug=AugmentConfig(bed_prob=0.0))
    by_region: dict[str, int] = {}  # first (lowest) reading-order index seen per region
    for ln in sorted(lines, key=lambda r: r.order):
        by_region.setdefault(ln.region, ln.order)
    if "page_number" in by_region and "body" in by_region:
        assert by_region["page_number"] < by_region["body"]
    if "header" in by_region and "body" in by_region:
        assert by_region["header"] < by_region["body"]
    if "marginalia" in by_region and "body" in by_region:
        assert by_region["marginalia"] > by_region["body"]
