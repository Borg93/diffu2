"""Tests for `--spec` — the CLI can drive the FULL PageSpec control surface from a JSON file (one or many),
so an agent/user reaches every knob (margin_keys, drop_cap, template, ...) without writing Python."""

from __future__ import annotations

import json
from pathlib import Path

from diffu_page.config import GenConfig
from diffu_page.generation import build_layouts, load_specs


def test_load_specs_single_object_and_array(tmp_path: Path):
    one = tmp_path / "one.json"
    one.write_text(json.dumps({"template": "landskap", "margin_keys": ["N:o 1"], "drop_cap": True}))
    specs = load_specs(one)
    assert len(specs) == 1
    assert specs[0].template == "landskap" and specs[0].margin_keys == ["N:o 1"] and specs[0].drop_cap

    many = tmp_path / "many.json"
    many.write_text(json.dumps([{"template": "letter"}, {"template": "register", "line_height": 44}]))
    specs = load_specs(many)
    assert [s.template for s in specs] == ["letter", "register"]


def test_build_layouts_respects_full_spec(tmp_path: Path):
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"template": "single_column", "drop_cap": True, "line_height": 40}))
    layouts = build_layouts(
        GenConfig(),
        n=3,
        seed=0,
        paragraphs=["Anno domini sextonhundra tjugusju forhordes menige man om skatten"] * 5,
        xml=None,
        measure=None,
        specs=load_specs(spec),
    )
    assert len(layouts) == 3
    # the full spec's drop_cap took effect on every page (an enlarged 'initial' box exists)
    assert all(any(b.region == "initial" for b in lay.boxes) for lay in layouts)
    assert all(lay.spec.template == "single_column" for lay in layouts)
