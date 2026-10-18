"""Tests for the YAML generation config + CLI override merge.

These guard the config contract a corpus run depends on: a YAML file maps onto the typed model (including
the nested augment block), a typo is rejected loudly rather than silently ignored, and a CLI flag overrides
the file only when it is actually given.
"""

from __future__ import annotations

import textwrap

import pytest
from diffu_page.config import GenConfig, Renderer, load_gen_config
from diffu_page.generation import build_layouts, with_overrides
from pydantic import ValidationError

_PARAS = ["the quick brown fox jumps over the lazy dog again and again"] * 6


def test_yaml_maps_onto_the_model_including_nested_augment(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        textwrap.dedent("""
        width: 1000
        height: 1400
        renderer: diffu
        binding_prob: 0.9
        augment:
          bed_prob: 0.3
          clasp_prob: 0.0
        """),
        encoding="utf-8",
    )
    cfg = load_gen_config(path)
    assert (cfg.width, cfg.height) == (1000, 1400)
    assert cfg.renderer is Renderer.diffu
    assert cfg.binding_prob == 0.9
    assert cfg.augment.bed_prob == 0.3 and cfg.augment.clasp_prob == 0.0


def test_unknown_key_is_rejected_not_ignored(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("widht: 1000\n", encoding="utf-8")  # a silent typo would waste a whole corpus run
    with pytest.raises(ValidationError):
        load_gen_config(path)


def test_out_of_range_value_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("binding_prob: 1.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_gen_config(path)


def test_no_path_returns_defaults():
    assert load_gen_config(None) == GenConfig()


_NO_OVERRIDES = dict(
    renderer=None, ckpt=None, style=None, paper_dir=None, binding_prob=None, bed_prob=None, clasp_prob=None
)


def test_cli_flags_override_only_where_set():
    cfg = GenConfig(binding_prob=0.4, augment={"bed_prob": 0.7})
    out = with_overrides(
        cfg, **{**_NO_OVERRIDES, "renderer": Renderer.diffu, "binding_prob": 0.1, "bed_prob": 0.0}
    )
    assert out.renderer is Renderer.diffu  # overridden
    assert out.binding_prob == 0.1  # overridden
    assert out.augment.bed_prob == 0.0  # nested override applied
    assert out.augment.clasp_prob == cfg.augment.clasp_prob  # untouched — its flag was None


def test_no_overrides_leaves_the_config_unchanged():
    cfg = GenConfig(width=900)
    assert with_overrides(cfg, **_NO_OVERRIDES) == cfg


def _regions(furniture_prob: float) -> set[str]:
    cfg = GenConfig(width=760, height=1040, furniture_prob=furniture_prob)
    layout = build_layouts(cfg, n=1, seed=1, paragraphs=_PARAS, xml=None, measure=None)[0]
    return {b.region for b in layout.boxes}


def test_furniture_prob_one_adds_header_marginalia_and_page_number():
    assert {"header", "marginalia", "page_number"} <= _regions(1.0)


def test_furniture_prob_zero_leaves_a_bare_body_page():
    assert not ({"header", "marginalia", "page_number"} & _regions(0.0))


def test_style_pool_gives_per_page_writer_variation():
    from pathlib import Path

    from diffu_page.generation import page_box_styles
    from diffu_page.layout import PageSpec, build_layout

    cfg = GenConfig(styles=[Path("a.png"), Path("b.png"), Path("c.png")])
    layouts = [build_layout(PageSpec(width=800, height=1100), _PARAS, seed=s) for s in range(12)]
    styles = page_box_styles(layouts, cfg, seed=0)
    used = {s for page in styles for s in page}
    assert used and used <= {"a.png", "b.png", "c.png"}, "assigned a style outside the pool"
    assert len({page[0] for page in styles if page}) > 1, "every page got the same writer (no variation)"


def test_single_style_is_used_for_every_box():
    from pathlib import Path

    from diffu_page.generation import page_box_styles
    from diffu_page.layout import PageSpec, build_layout

    cfg = GenConfig(style=Path("only.png"))
    layouts = [build_layout(PageSpec(width=800, height=1100, marginalia=["m"]), _PARAS, seed=1)]
    assert set(page_box_styles(layouts, cfg, seed=1)[0]) == {"only.png"}


def test_split_fractions_over_one_are_rejected():
    with pytest.raises(ValidationError):
        GenConfig(val_frac=0.7, test_frac=0.7)


def test_semi_table_marks_headers_and_row_numbers_as_printed():
    # typeset_furniture now defaults to False (handwritten furniture — most corpus registers are fully
    # handwritten); printed headers/row numbers are the opt-in PRE-PRINTED-form look tested here.
    from diffu_page.layout import PageSpec, build_layout

    spec = PageSpec(width=900, height=1100, template="semi_table", typeset_furniture=True)
    layout = build_layout(spec, ["alpha beta", "gamma delta", "epsilon zeta"] * 4, seed=1)
    printed = {b.region for b in layout.boxes if b.printed}
    assert {"table_header", "row_number"} <= printed
    # handwriting cells must NOT be flagged printed (they'd render as typeset otherwise)
    assert all(not b.printed for b in layout.boxes if b.region == "table")


def test_semi_table_furniture_handwritten_by_default():
    from diffu_page.layout import PageSpec, build_layout

    layout = build_layout(
        PageSpec(width=900, height=1100, template="semi_table"),
        ["alpha beta", "gamma delta", "epsilon zeta"] * 4,
        seed=1,
    )
    assert all(not b.printed for b in layout.boxes)  # nothing typeset unless typeset_furniture=True
