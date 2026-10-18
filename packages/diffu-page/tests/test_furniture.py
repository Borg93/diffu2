"""Tests for high-frequency page furniture the corpus study flagged: catchword (custos) + underline
emphasis. Both must travel to the PAGE-XML ground truth (a catch-word region / an underlined TextStyle).
"""

from __future__ import annotations

import numpy as np
from diffu_page.augment import AugmentConfig
from diffu_page.compose import compose_page
from diffu_page.labels import PageLabels, to_page_xml
from diffu_page.layout import PageSpec, build_layout
from diffu_page.render import PlaceholderRenderer

_SENT = "Goteborgs poliskammare den fjarde maj artonhundra attiotva"
_FLAT = AugmentConfig(bed_prob=0.0, microfilm_prob=0.0, skew=0.0, rotate=(0.0, 0.0), stamp_prob=0.0)


def test_catchword_sits_at_the_foot_outer_corner():
    spec = PageSpec(width=1200, height=1700, catchword="Anno", recto=True)
    layout = build_layout(spec, [_SENT] * 4, seed=1)
    cw = [b for b in layout.boxes if b.region == "catchword"]
    assert len(cw) == 1 and cw[0].text == "Anno", "one catchword box carrying the word"
    assert cw[0].y > spec.height * 0.75, "the catchword sits at the foot of the page"
    assert cw[0].x + cw[0].w > spec.width * 0.6, "a recto catchword sits at the outer (right) corner"
    verso = build_layout(
        PageSpec(width=1200, height=1700, catchword="Anno", recto=False), [_SENT] * 4, seed=1
    )
    vb = next(b for b in verso.boxes if b.region == "catchword")
    assert vb.x < spec.width * 0.4, "a verso catchword sits at the outer (left) corner"


def test_catchword_exports_as_a_page_catch_word_region():
    spec = PageSpec(width=1200, height=1700, catchword="Anno")
    _, lines = compose_page(build_layout(spec, [_SENT] * 4, seed=1), PlaceholderRenderer(), seed=1, aug=_FLAT)
    xml = to_page_xml(PageLabels(width=spec.width, height=spec.height, lines=lines))
    assert "catch-word" in xml, "the catchword must export as a PAGE catch-word region"


def test_underline_tags_lines_and_exports_the_underlined_style():
    spec = PageSpec(width=1200, height=1700, template="single_column", line_height=40, underline_prob=1.0)
    layout = build_layout(spec, [" ".join([_SENT] * 3)] * 4, seed=1)
    assert [b for b in layout.boxes if b.tag == "underline"], "underline_prob=1 must tag body lines"
    _, lines = compose_page(layout, PlaceholderRenderer(), seed=1, aug=_FLAT)
    assert any(ln.tag == "underline" for ln in lines), "the underline tag travels to the label"
    xml = to_page_xml(PageLabels(width=spec.width, height=spec.height, lines=lines))
    assert 'underlined="true"' in xml, "underline must export to a PAGE TextStyle"


def test_no_furniture_by_default():
    layout = build_layout(PageSpec(width=1000, height=1400), [_SENT] * 4, seed=1)
    assert not [b for b in layout.boxes if b.region == "catchword"], "no catchword unless spec.catchword set"
    assert not [b for b in layout.boxes if b.tag == "underline"], "no underline unless underline_prob > 0"


def test_scribal_marks_fire_and_are_label_safe():
    spec = PageSpec(width=1200, height=1700, template="single_column", line_height=40)
    layout = build_layout(
        spec, [" ".join([_SENT] * 3)] * 5, seed=2, measure=PlaceholderRenderer().measure_width
    )
    trace: dict[str, float] = {}
    _, lines = compose_page(
        layout,
        PlaceholderRenderer(),
        seed=2,
        aug=AugmentConfig(paraph_prob=1.0, margin_mark_prob=1.0),
        trace=trace,
    )
    assert trace.get("paraph") == 1.0 and trace.get("margin_mark") == 1.0, "forced scribal marks must trace"
    for ln in lines:  # photometric ink -> labels never move
        for x, y in ln.polygon:
            assert 0 <= x < spec.width and 0 <= y < spec.height, "a scribal mark moved a label off the page"


def test_rubric_headings_read_in_flow_position_and_enumerators_prefix_paragraphs():
    spec = PageSpec(
        width=1200,
        height=1700,
        template="single_column",
        line_height=38,
        section_headings=["Resolutio"],
        enum_prob=1.0,
    )
    layout = build_layout(
        spec, [" ".join([_SENT] * 3)] * 7, seed=3, measure=PlaceholderRenderer().measure_width
    )
    rubrics = [b for b in layout.boxes if b.region == "rubric"]
    assert rubrics and all(b.align == "center" for b in rubrics), "a centered rubric heading is inserted"
    assert [b for b in layout.boxes if b.region == "body" and b.text.startswith("§")], "enum prefixes a §. N."
    _, lines = compose_page(layout, PlaceholderRenderer(), seed=3, aug=_FLAT)
    ordered = sorted(lines, key=lambda ln: ln.order)
    rub_idx = next(i for i, ln in enumerate(ordered) if ln.region == "rubric")
    assert rub_idx > 0 and ordered[rub_idx - 1].region == "body", "a rubric reads in flow position, not first"


def test_scribal_marks_are_off_by_default_and_consume_no_rng_when_off():
    # opt-in: the vanilla config draws no paraph/margin mark AND stays byte-identical (short-circuit before rng)
    spec = PageSpec(width=900, height=1240)
    layout = build_layout(spec, [_SENT] * 4, seed=1)
    off = AugmentConfig(paraph_prob=0.0, margin_mark_prob=0.0)
    a, _ = compose_page(layout, PlaceholderRenderer(), seed=1, aug=off)
    # a config that differs ONLY in the (disabled) scribal knobs must produce a byte-identical page
    b, _ = compose_page(layout, PlaceholderRenderer(), seed=1, aug=AugmentConfig())
    assert np.array_equal(a, b), "disabled scribal knobs must not perturb the page (no rng consumed)"
