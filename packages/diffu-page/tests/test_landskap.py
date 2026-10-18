"""Tests for the boskapslängd / landskapshandling template — a sparse display-headed tax roll: display
section headers over indented label→leader→value rows, everything fitted in-bounds inside its slot."""

from __future__ import annotations

from diffu_page.compose import compose_page
from diffu_page.layout import PageSpec, build_layout
from diffu_page.render import PlaceholderRenderer
from diffu_page.validate import poly_to_box


def _compose(seed: int):
    spec = PageSpec(template="landskap", line_height=40, line_gap=0.55)
    renderer = PlaceholderRenderer()
    layout = build_layout(spec, ["unused"], seed=seed)
    page, lines = compose_page(layout, renderer, seed=seed)
    return spec, layout, page, lines


def test_landskap_has_display_headers_over_leader_rows():
    _, layout, _, _ = _compose(3)
    regions = {b.region for b in layout.boxes}
    assert "reg_header" in regions, "sections need a display header"
    assert {"reg_label", "reg_value"} <= regions, "entries are label -> value rows"
    # the header is rendered at a DISPLAY scale (taller than a body line)
    headers = [b for b in layout.boxes if b.region == "reg_header"]
    body_h = next(b.h for b in layout.boxes if b.region == "reg_label")
    assert any(h.h > body_h for h in headers), "section headers should be enlarged (display)"
    # dotted leaders bridge label to value
    assert any(r.style == "dashed" for r in layout.rules), "leader rules missing"


def test_landskap_values_and_labels_fit_their_slots():
    spec, _, page, lines = _compose(5)
    h, w = page.shape[:2]
    oob = sum(1 for ln in lines for x, y in ln.polygon if not (0 <= x < w and 0 <= y < h))
    assert oob == 0, f"{oob} label points outside the page"
    # a value cell (a small figure) never spills past its narrow column's natural slot
    for ln in lines:
        if ln.region == "reg_value":
            x0, _, x1, _ = poly_to_box(ln.polygon)
            assert x1 - x0 <= 4 * spec.line_height, f"value {ln.text!r} too wide ({x1 - x0}px)"


def test_landskap_reads_top_to_bottom():
    _, _, _, lines = _compose(7)
    ordered = sorted(lines, key=lambda ln: ln.order)
    assert [ln.order for ln in ordered] == list(range(len(ordered))), "order must be a contiguous permutation"
