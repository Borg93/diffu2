"""Tests for the enlarged opening initial (versal) — the first letter becomes its own enlarged, labelled
``initial`` box; the opening line loses that letter and shifts right; the GT stays exact and reads in flow."""

from __future__ import annotations

from diffu_page.compose import compose_page
from diffu_page.layout import PageSpec, build_layout
from diffu_page.render import PlaceholderRenderer

_PARAS = ["Anno domini sextonhundra tjugusju forhordes menige man om skatten", "en annan paragraf har"]


def test_drop_cap_enlarges_the_opening_initial():
    spec = PageSpec(template="single_column", line_height=40, drop_cap=True)
    layout = build_layout(spec, _PARAS, seed=0)
    initials = [b for b in layout.boxes if b.region == "initial"]
    assert len(initials) == 1, "exactly one opening initial"
    cap = initials[0]
    body_h = next(b.h for b in layout.boxes if b.region == "body")
    assert cap.h > body_h, "the initial must be enlarged vs a body line"
    assert cap.text == "A", f"the initial is the opening letter, got {cap.text!r}"
    # the opening body line lost its first letter and shifted right to clear the initial
    opener = min((b for b in layout.boxes if b.region == "body"), key=lambda b: (b.y, b.x))
    assert opener.text.startswith("nno domini"), opener.text
    assert opener.x > cap.x, "the opening line shifts right of the initial"


def test_no_drop_cap_by_default():
    layout = build_layout(PageSpec(template="single_column", line_height=40), _PARAS, seed=0)
    assert not any(b.region == "initial" for b in layout.boxes)


def test_drop_cap_reads_first_and_labels_are_exact():
    spec = PageSpec(template="single_column", line_height=40, drop_cap=True)
    renderer = PlaceholderRenderer()
    layout = build_layout(spec, _PARAS, seed=1, measure=renderer.measure_width)
    _, lines = compose_page(layout, renderer, seed=1)
    ordered = sorted(lines, key=lambda ln: ln.order)
    initial = next(ln for ln in ordered if ln.region == "initial")
    # the initial reads immediately before the remainder of its opening line (flow position, not last)
    i = ordered.index(initial)
    assert ordered[i + 1].region == "body", "initial should read right before its body line"
    # reconstructed opening = initial + remainder = the original first word intact
    assert (initial.text + ordered[i + 1].text).startswith("Anno domini"), (initial.text, ordered[i + 1].text)
    assert [ln.order for ln in ordered] == list(range(len(ordered)))
