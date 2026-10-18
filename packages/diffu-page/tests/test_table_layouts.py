"""Tests for the tabular / register layouts (register, ledger, semi_table) and the single-leaf spread
split on XML replay — the structures the GT-Lejonet corpus is full of and a prose template can't produce.
All run on the CPU PlaceholderRenderer.
"""

from __future__ import annotations

import numpy as np
import pytest
from diffu_page.augment import AugmentConfig
from diffu_page.compose import compose_page
from diffu_page.labels import LineLabel
from diffu_page.layout import Layout, LineBox, PageSpec, Rule, build_layout, to_single_leaf
from diffu_page.render import PlaceholderRenderer
from diffu_page.validate import validate_table

_NAMES = ["Anders Persson", "Erik Nilsson", "Johan Bergstrom", "Per Olsson", "Sven Larsson", "Nils Andersson"]
_FLAT = AugmentConfig(bed_prob=0.0, microfilm_prob=0.0, skew=0.0, rotate=(0.0, 0.0))


def _register_layout(seed: int = 1):
    rows = ["Vreta Kloster Bonder"] + [f"{_NAMES[i % len(_NAMES)]}: {i + 1}" for i in range(6)]
    rows += ["Fralse Bonder"] + [f"{_NAMES[i % len(_NAMES)]}: {i + 2}" for i in range(5)]
    spec = PageSpec(width=1200, height=1700, template="register", line_height=42)
    return spec, build_layout(spec, rows, seed=seed)


def test_register_is_a_leader_dot_register():
    _, layout = _register_layout()
    regions = {b.region for b in layout.boxes}
    assert {"reg_header", "reg_label", "reg_value"} <= regions, "register needs headers + label/value entries"
    assert layout.rules, "each entry must carry a leader rule bridging label -> value"
    values = [b for b in layout.boxes if b.region == "reg_value"]
    assert values and all(b.align == "right" for b in values), "register values are right-aligned"
    # every leader rule is horizontal and sits between the label column and the value column
    for r in layout.rules:
        assert r.y1 == r.y2 and r.x2 > r.x1, "a leader rule must be a horizontal segment"


def test_ledger_has_numeric_columns_and_subtotals():
    spec = PageSpec(width=1400, height=2000, template="ledger", line_height=38, table_cols=5)
    text = [" ".join(n.split()[0] for n in _NAMES) for _ in range(40)]
    layout = build_layout(spec, text, seed=1)
    regions = {b.region for b in layout.boxes}
    assert "table_header" in regions and "table" in regions, "a ledger has a header row + body cells"
    verticals = [r for r in layout.rules if r.x1 == r.x2]
    assert len(verticals) >= 3, "a numeric ledger needs vertical column rules"
    assert any(b.text == "Summa" for b in layout.boxes), "a ledger must emit subtotal (Summa) rows"
    numbers = [b for b in layout.boxes if b.region == "table" and b.align == "right"]
    assert numbers, "numeric cells are right-aligned"


def test_semi_table_is_a_typed_grid_with_money_columns_and_totals():
    spec = PageSpec(
        width=1654, height=2339, template="semi_table", line_height=40, table_rows=18, table_cols=7
    )
    layout = build_layout(spec, ["hemman skatte Anders Persson Erik Landvetter socken krono"], seed=1)
    assert any(b.text == "Summa" for b in layout.boxes), "a money grid emits a Summa totals row"
    assert [b for b in layout.boxes if b.region == "table" and b.align == "right"], (
        "money columns are right-aligned"
    )
    verticals = sorted({r.x1 for r in layout.rules if r.x1 == r.x2})
    gaps = [b - a for a, b in zip(verticals, verticals[1:], strict=False)]
    assert max(gaps) > 1.5 * min(gaps), "columns are proportioned (non-uniform), not a uniform grid"


@pytest.mark.parametrize(
    "template,kw",
    [
        ("register", {}),
        ("ledger", {"table_cols": 5}),
        ("semi_table", {"table_rows": 16, "table_cols": 6}),
    ],
)
def test_table_templates_seat_labels_in_bounds_and_reproducibly(template: str, kw: dict):
    spec = PageSpec(width=1200, height=1700, template=template, line_height=40, **kw)
    text = ["Vreta Bonder"] + [f"{_NAMES[i % len(_NAMES)]}: {i}" for i in range(24)]
    layout = build_layout(spec, text, seed=2)
    page_a, lines_a = compose_page(layout, PlaceholderRenderer(), seed=2, aug=_FLAT)
    page_b, lines_b = compose_page(layout, PlaceholderRenderer(), seed=2, aug=_FLAT)
    assert lines_a, f"{template} produced no labels"
    assert np.array_equal(page_a, page_b), f"{template} is not reproducible"
    assert [ln.model_dump() for ln in lines_a] == [ln.model_dump() for ln in lines_b], (
        f"{template} labels differ"
    )
    for ln in lines_a:
        for x, y in ln.polygon:
            assert 0 <= x < spec.width and 0 <= y < spec.height, (
                f"{template} label left the page at ({x},{y})"
            )


def test_single_leaf_split_keeps_one_side_of_a_spread():
    # a synthetic two-column spread: 5 lines on the left half, 5 on the right half of a wide page
    spec = PageSpec(width=4000, height=2000)
    boxes = [
        LineBox(text=f"L{i}", x=200, y=100 + i * 120, w=1400, h=80, baseline_y=160 + i * 120)
        for i in range(5)
    ]
    boxes += [
        LineBox(text=f"R{i}", x=2300, y=100 + i * 120, w=1400, h=80, baseline_y=160 + i * 120)
        for i in range(6)
    ]
    leaf = to_single_leaf(Layout(spec=spec, boxes=boxes), prefer="denser")
    assert leaf.spec.width < spec.width, "the single leaf must be narrower than the full spread"
    assert len(leaf.boxes) == 6, "the denser (right) leaf has 6 lines"
    assert all(b.text.startswith("R") for b in leaf.boxes), "only the right leaf's lines are kept"
    assert min(b.x for b in leaf.boxes) >= 0, "the kept leaf is re-origined onto the new canvas"
    assert max(b.x + b.w for b in leaf.boxes) <= leaf.spec.width, "kept lines fit the new single-leaf width"


def test_blank_leaf_is_near_empty_with_page_furniture():
    spec = PageSpec(width=1200, height=1700, template="blank_leaf", page_number="7")
    layout = build_layout(spec, [" ".join(_NAMES)] * 3, seed=1)
    body = [b for b in layout.boxes if b.region == "body"]
    assert len(body) <= 4, "a blank leaf carries at most a few stray lines"
    assert any(b.region == "page_number" for b in layout.boxes), "page furniture (number) still present"


def test_single_leaf_prefer_blank_keeps_the_emptier_half():
    spec = PageSpec(width=4000, height=2000)
    boxes = [
        LineBox(text=f"L{i}", x=200, y=100 + i * 120, w=1400, h=80, baseline_y=160 + i * 120)
        for i in range(8)
    ]
    boxes += [
        LineBox(text=f"R{i}", x=2300, y=100 + i * 120, w=1400, h=80, baseline_y=160 + i * 120)
        for i in range(3)
    ]
    leaf = to_single_leaf(Layout(spec=spec, boxes=boxes), prefer="blank")
    assert all(b.text.startswith("R") for b in leaf.boxes), "prefer='blank' keeps the emptier (3-line) leaf"


def test_single_leaf_leaves_a_genuine_single_page_untouched():
    spec = PageSpec(width=1600, height=2000)
    boxes = [
        LineBox(text=f"L{i}", x=200, y=100 + i * 120, w=1200, h=80, baseline_y=160 + i * 120)
        for i in range(8)
    ]
    leaf = to_single_leaf(Layout(spec=spec, boxes=boxes))
    assert leaf.spec.width == spec.width and len(leaf.boxes) == 8, "a single page must pass through unchanged"


@pytest.mark.parametrize(
    "template,kw",
    [
        ("register", {}),
        ("ledger", {"table_cols": 5}),
        ("semi_table", {"table_rows": 16, "table_cols": 6}),
    ],
)
def test_validate_table_accepts_a_well_formed_table(template: str, kw: dict):
    spec = PageSpec(width=1654, height=2339, template=template, line_height=42, **kw)
    text = ["Vreta Bonder"] + [f"{_NAMES[i % len(_NAMES)]}: {i}" for i in range(20)]
    layout = build_layout(spec, text, seed=1)
    _, lines = compose_page(layout, PlaceholderRenderer(), seed=1, aug=_FLAT)
    assert validate_table(layout, lines) == [], f"{template} should be structurally correct"


def test_validate_table_rejects_overlapping_cells_and_rule_crossing():
    spec = PageSpec(width=1000, height=1400)
    rules = [Rule(x1=500, y1=0, x2=500, y2=1400)]  # one vertical column rule
    lines = [
        LineLabel(  # ink straddles the column rule at x=500
            text="crosses",
            polygon=[(460, 100), (560, 100), (560, 140), (460, 140)],
            baseline=[(460, 135), (560, 135)],
            region="table",
        ),
        LineLabel(text="A", polygon=[(100, 200), (300, 200), (300, 240), (100, 240)], region="table"),
        LineLabel(text="B", polygon=[(150, 210), (320, 210), (320, 250), (150, 250)], region="table"),
    ]
    problems = validate_table(Layout(spec=spec, boxes=[], rules=rules), lines)
    assert any("crosses column" in p for p in problems), "must flag a cell crossing a column rule"
    assert any("overlap" in p for p in problems), "must flag two overlapping cells"


def test_register_reads_row_major_label_then_value():
    spec = PageSpec(width=1200, height=1700, template="register", line_height=44, line_gap=0.5)
    rows = ["Vreta Bonder"] + [f"{_NAMES[i % len(_NAMES)]}: {i + 1}" for i in range(4)]
    layout = build_layout(spec, rows, seed=1)
    _, lines = compose_page(layout, PlaceholderRenderer(), seed=1, aug=_FLAT)
    seq = [ln.region for ln in sorted(lines, key=lambda ln: ln.order)]
    assert seq[0] == "reg_header", "the section header reads first"
    # after the header, labels and values must interleave (label, value, label, value ...), not group
    body = seq[1:]
    assert body[::2] == ["reg_label"] * len(body[::2]), f"labels must precede their values row by row: {seq}"
    assert body[1::2] == ["reg_value"] * len(body[1::2]), f"each value follows its label: {seq}"
    assert validate_table(layout, lines) == [], "a correct register passes reading-order validation"


def test_validate_table_rejects_a_grouped_register_reading_order():
    # the old bug: all values, then all labels (right-to-left within a row) -> must be flagged
    spec = PageSpec(width=1000, height=1400)
    lines = [
        LineLabel(
            text="1",
            order=0,
            polygon=[(600, 100), (650, 100), (650, 140), (600, 140)],
            baseline=[(600, 135), (650, 135)],
            region="reg_value",
        ),
        LineLabel(
            text="Anders",
            order=1,
            polygon=[(100, 100), (300, 100), (300, 140), (100, 140)],
            baseline=[(100, 135), (300, 135)],
            region="reg_label",
        ),
    ]
    problems = validate_table(Layout(spec=spec, boxes=[], rules=[]), lines)
    assert any("right-to-left" in p or "jumps up" in p for p in problems), "must flag a non-row-major order"
