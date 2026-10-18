"""Tests for label/XML correctness — a correctly composed page passes both validators, and each fault
(out-of-bounds polygon, escaping word box, broken reading order, dropped XML line) is caught."""

from __future__ import annotations

from diffu_page.compose import compose_page
from diffu_page.labels import LineLabel, PageLabels, WordLabel
from diffu_page.layout import PageSpec, build_layout
from diffu_page.render import PlaceholderRenderer
from diffu_page.validate import validate_labels, validate_xml_roundtrip


def _composed_page() -> PageLabels:
    spec = PageSpec(width=1654, height=2339, template="single_column", line_height=40)
    renderer = PlaceholderRenderer()
    text = ["Goteborgs poliskammare den fjarde maj anhallen for fylleri och forargelse"] * 20
    layout = build_layout(spec, text, seed=0, measure=renderer.measure_width)
    page, lines = compose_page(layout, renderer, seed=0)
    h, w = page.shape[:2]
    return PageLabels(width=w, height=h, image="page.png", lines=lines)


def test_a_correctly_composed_page_has_valid_labels():
    labels = _composed_page()
    assert labels.lines, "the page should have lines"
    assert validate_labels(labels) == [], "a real composed page must have geometrically valid labels"


def test_a_correctly_composed_page_round_trips_to_every_format():
    labels = _composed_page()
    assert validate_xml_roundtrip(labels) == [], "PAGE/ALTO/JSON export must be lossless + well-formed"


def test_out_of_bounds_polygon_is_caught():
    labels = PageLabels(
        width=100,
        height=100,
        lines=[LineLabel(text="x", polygon=[(10, 10), (200, 10), (200, 30), (10, 30)], order=0)],
    )
    problems = validate_labels(labels)
    assert any("leaves the page" in p for p in problems), problems


def test_word_centroid_off_its_line_is_caught():
    line = LineLabel(  # single word so the transcription reconstructs — isolates the centroid check
        text="alpha",
        polygon=[(10, 10), (90, 10), (90, 30), (10, 30)],
        words=[WordLabel(text="alpha", polygon=[(120, 12), (160, 12), (160, 28), (120, 28)])],
        order=0,
    )
    problems = validate_labels(PageLabels(width=500, height=500, lines=[line]))
    assert any("centroid sits off" in p for p in problems), problems


def test_word_transcriptions_must_reconstruct_the_line():
    line = LineLabel(
        text="one two",
        polygon=[(10, 10), (90, 10), (90, 30), (10, 30)],
        words=[WordLabel(text="one", polygon=[(12, 12), (40, 12), (40, 28), (12, 28)])],  # 'two' missing
        order=0,
    )
    problems = validate_labels(PageLabels(width=500, height=500, lines=[line]))
    assert any("reconstruct" in p for p in problems), problems


def test_broken_reading_order_is_caught():
    labels = PageLabels(
        width=500,
        height=500,
        lines=[
            LineLabel(text="a", polygon=[(10, 10), (90, 10), (90, 30), (10, 30)], order=0),
            LineLabel(text="b", polygon=[(10, 40), (90, 40), (90, 60), (10, 60)], order=0),  # dup order
        ],
    )
    problems = validate_labels(labels)
    assert any("permutation" in p for p in problems), problems
