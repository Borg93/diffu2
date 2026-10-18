"""Tests for keyed marginal notes — each note is seated beside its paragraph and reads INTERLEAVED with the
body at that point (not appended last like an unkeyed running marginalia note)."""

from __future__ import annotations

from diffu_page.compose import compose_page
from diffu_page.layout import PageSpec, build_layout
from diffu_page.render import PlaceholderRenderer

_PARAS = [
    "den fjarde maj artonhundra attiotva forhordes arbetskarlen Anders Persson",
    "vittnen instamde och saken hanskots till rattens vidare provning i staden",
    "en summa af tolf riksdaler femtio ore utdomdes i boter och rattegangskostnad",
]


def _compose(spec: PageSpec):
    renderer = PlaceholderRenderer()
    layout = build_layout(spec, _PARAS, seed=0, measure=renderer.measure_width)
    _, lines = compose_page(layout, renderer, seed=0)
    return lines


def test_keyed_notes_read_interleaved_with_the_body():
    spec = PageSpec(template="single_column", line_height=40, margin_keys=["N:o 1", "N:o 2", "N:o 3"])
    lines = _compose(spec)
    keys = [ln for ln in lines if ln.region == "margin_key"]
    assert len(keys) == 3, f"expected one key per paragraph, got {len(keys)}"
    ordered = sorted(lines, key=lambda ln: ln.order)
    # each key reads immediately before a body line (its paragraph's first line), never all bunched last
    for key in keys:
        i = ordered.index(key)
        assert i + 1 < len(ordered), f"key {key.text!r} is last — not interleaved"
        assert ordered[i + 1].region == "body", f"key {key.text!r} not followed by its body line"
    # the keys are not all pushed to the tail (as unkeyed marginalia would be)
    assert max(ordered.index(k) for k in keys) < len(ordered) - 1


def test_no_margin_keys_leaves_reading_order_unchanged():
    lines = _compose(PageSpec(template="single_column", line_height=40))
    assert not any(ln.region == "margin_key" for ln in lines)
    assert [ln.order for ln in sorted(lines, key=lambda ln: ln.order)] == list(range(len(lines)))
