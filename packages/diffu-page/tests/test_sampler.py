"""Tests for the corpus-proportion layout sampler — the empirical draw matches the measured mix, every
kind builds a valid layout, and the mix is reweightable."""

from __future__ import annotations

import random
from collections import Counter

from diffu_page.layout import build_layout, validate_layout
from diffu_page.sampler import DEFAULT_MIX, SPREAD_KINDS, LayoutMix, sample_kind, sample_page_spec

_TEMPLATE_NAMES = {
    "single_column",
    "letter",
    "blank_leaf",
    "register",
    "ledger",
    "semi_table",
    "landskap",
    "spread",
}
_TEXT = ["Goteborgs poliskammare den fjarde maj anhallen for fylleri"] * 12


def test_sampler_matches_the_corpus_proportions():
    rng = random.Random(0)
    n = 5000
    counts = Counter(sample_kind(rng) for _ in range(n))
    for kind, target in DEFAULT_MIX.weights.items():
        frac = counts[kind] / n
        assert abs(frac - target) < 0.03, f"{kind}: sampled {frac:.3f} vs target {target:.3f}"


def test_every_kind_builds_a_valid_layout():
    rng = random.Random(1)
    seen: set[str] = set()
    for _ in range(300):
        kind, spec = sample_page_spec(rng)
        seen.add(kind)
        assert spec.template in _TEMPLATE_NAMES, f"{kind} -> unknown template {spec.template}"
        layout = build_layout(spec, _TEXT, seed=0)  # must not raise
        assert layout.boxes or spec.template == "blank_leaf", f"{kind} produced no boxes"
    assert seen == set(DEFAULT_MIX.weights), "all kinds should be sampled over 300 draws"


def test_layout_mix_is_reweightable():
    mix = LayoutMix(weights={"letter": 1.0, "blank": 1.0})
    rng = random.Random(2)
    kinds = {sample_kind(rng, mix) for _ in range(50)}
    assert kinds <= {"letter", "blank"}, "a custom mix only draws its own kinds"


def test_spread_frac_zero_yields_only_single_leaves():
    mix = LayoutMix(spread_frac=0.0)
    rng = random.Random(3)
    for _ in range(300):
        _, spec = sample_page_spec(rng, mix)
        assert not spec.spread, "spread_frac=0 must never emit a spread"
        assert spec.height > spec.width, "a single leaf stays portrait"


def test_spread_frac_one_spreads_every_eligible_kind():
    mix = LayoutMix(spread_frac=1.0)
    rng = random.Random(4)
    spread_kinds_seen: set[str] = set()
    for _ in range(400):
        kind, spec = sample_page_spec(rng, mix)
        if kind in SPREAD_KINDS:
            spread_kinds_seen.add(kind)
            assert spec.spread, f"eligible kind {kind} must spread at spread_frac=1"
            assert spec.width > spec.height, "a spread canvas is landscape (two leaves + gutter)"
            assert not spec.binding, "a spread's spine is the central gutter, never a single-edge binding"
            assert spec.page_number is None, "_spread writes its own verso/recto corner numbers"
        else:
            # the spread template flows prose-like paragraphs only; tabular kinds stay single-leaf
            assert not spec.spread, f"tabular kind {kind} must stay single-leaf"
    assert spread_kinds_seen == set(SPREAD_KINDS), "every eligible kind should spread over 400 draws"


def test_sampled_spreads_build_valid_two_leaf_layouts():
    mix = LayoutMix(spread_frac=1.0)
    rng = random.Random(5)
    checked = both_leaves = 0
    while checked < 12:
        _, spec = sample_page_spec(rng, mix)
        if not spec.spread:
            continue
        layout = build_layout(spec, _TEXT, seed=checked)
        assert layout.boxes, "a spread must lay out at least one leaf"
        assert validate_layout(layout) == [], "a spread layout must be in-bounds and overlap-free"
        cx = spec.width // 2
        gutter = max(spec.line_height, int(0.03 * spec.width))  # templates._spread's half-gutter
        assert all(b.x + b.w <= cx - gutter or b.x >= cx + gutter for b in layout.boxes), (
            "no box may intrude into the central gutter band"
        )
        left = any(b.x + b.w / 2 < cx for b in layout.boxes)
        right = any(b.x + b.w / 2 >= cx for b in layout.boxes)
        both_leaves += left and right
        checked += 1
    assert both_leaves >= 6, "most spreads carry text on BOTH leaves (~15% blank one side)"
