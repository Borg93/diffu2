"""Core tests — the gradio-free API. No model weights or GPU: heavy paths are monkeypatched so the
architecture guard, engine caching, and config round-trips are all verified in milliseconds.
"""

from __future__ import annotations

import pytest
from diffu_page.config import GenConfig
from diffu_studio import checkpoints as ckpt_mod
from diffu_studio import engine as engine_mod
from diffu_studio.checkpoints import ArchConfig, CheckpointMismatch, discover_checkpoints, resolve_weights
from diffu_studio.configs import (
    SampleSettings,
    build_augment,
    build_gen_config,
    default_configs,
    list_formats,
    list_templates,
)
from diffu_studio.device import DeviceKind, resolve_device
from diffu_studio.engine import StudioEngine


def test_config_round_trips():
    cfg = build_gen_config(template="spread", line_height=48, steps=30, cfg_scale=6.0)
    assert cfg.template.value == "spread"
    assert cfg.line_height == 48
    assert cfg.steps == 30
    # a GenConfig re-validates from its own JSON dump (the API contract for the control panel)
    assert GenConfig.model_validate(cfg.model_dump()).line_height == 48


def test_build_augment_maps_curated_controls_to_ranges_and_ink():
    aug = build_augment(rotate_max=4.0, wash_max=0.5, ink_fade_max=0.2, ink_color="#20304a")
    assert aug.rotate == (0.0, 4.0)
    assert aug.wash == (0.0, 0.5)
    assert aug.ink_fade == (0.0, 0.2)
    # "#20304a" (R=0x20, G=0x30, B=0x4a) -> a single BGR ink in [0,1]
    assert aug.ink_colors == [(0x4A / 255, 0x30 / 255, 0x20 / 255)]


def test_build_augment_blank_ink_keeps_default_palette():
    assert len(build_augment(ink_color="").ink_colors) > 1


def test_sample_settings_validates_ranges():
    with pytest.raises(ValueError):
        SampleSettings(steps=0)
    with pytest.raises(ValueError):
        SampleSettings(cfg_scale=99.0)


def test_default_configs_and_lists():
    cfgs = default_configs()
    assert set(cfgs) == {"gen", "augment", "templates", "formats"}
    assert "single_column" in list_templates()
    assert set(list_formats()) == {"json", "page", "alto"}


def test_resolve_device_cpu_is_non_interactive():
    dev = resolve_device("cpu")
    assert dev.kind is DeviceKind.cpu
    assert dev.torch_device == "cpu"
    assert dev.interactive is False


def test_discover_checkpoints_orders_newest_trained_first(tmp_path):
    import os

    # (run, step, mtime): a STALE run with a higher step must NOT outrank the freshly-trained one.
    for run, step, mtime in [("stale", 400000, 100), ("active", 80000, 300), ("active", 40000, 200)]:
        weights = tmp_path / run / f"step_{step}" / "model.safetensors"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"")
        os.utime(weights, (mtime, mtime))
    found = discover_checkpoints(str(tmp_path))
    assert [c.label for c in found] == ["active/step_80000", "active/step_40000", "stale/step_400000"]


def test_resolve_weights_prefers_ema(tmp_path):
    d = tmp_path / "step_1000"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"")
    (d / "ema_model.safetensors").write_bytes(b"")
    assert resolve_weights(str(d)).endswith("ema_model.safetensors")


def test_load_model_wraps_arch_mismatch_as_checkpoint_mismatch(monkeypatch):
    # Stand in for the heavy model build; only the guard's behaviour is under test.
    class _FakeModel:
        def to(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

    def _raise(_model, _ckpt):
        raise RuntimeError("145/147 conditioner weights absent")

    monkeypatch.setattr("diffu.model.Diffu", lambda _cfg: _FakeModel())
    monkeypatch.setattr("diffu.generate.load_checkpoint", lambda _m, _p: None)
    monkeypatch.setattr("diffu.generate.assert_conditioner_loaded", _raise)

    with pytest.raises(CheckpointMismatch, match="conditioner"):
        ckpt_mod.load_model("does-not-matter.safetensors", ArchConfig(), resolve_device("cpu"))


def _placeholder_cache(width=640, height=480):
    from diffu_page.layout import PageSpec, build_layout
    from diffu_page.render import PlaceholderRenderer, render_boxes
    from diffu_studio.configs import build_gen_config
    from diffu_studio.steps import RenderCache

    paragraphs = ["hello world foo bar", "a second line here"]
    layout = build_layout(
        PageSpec(width=width, height=height), paragraphs, seed=1
    )  # no measure → px_per_char
    inks = render_boxes(layout.boxes, PlaceholderRenderer())  # stand-in for model inks — no GPU
    gen = build_gen_config(width=width, height=height, formats=["page", "alto", "json"])
    return RenderCache(layout=layout, inks=inks, seed=1, gen=gen, paragraphs=paragraphs, measure=None)


def test_recompose_page_reuses_cached_inks_and_only_changes_the_scan_look():
    import numpy as np
    from diffu_page.augment import AugmentConfig
    from diffu_studio.page import recompose_page

    cache = _placeholder_cache()
    plain = recompose_page(cache, AugmentConfig(bed_prob=0.0, microfilm_prob=0.0), ["json", "page"])
    filmed = recompose_page(cache, AugmentConfig(bed_prob=0.0, microfilm_prob=1.0), ["json"])

    assert plain.phase == "final" and plain.page is not None
    assert set(plain.labels or {}) == {"json", "page"} and "<TextLine" in (plain.labels or {})["page"]
    assert plain.page_labels is not None and len(plain.page_labels.lines) >= 1
    assert not np.array_equal(np.asarray(plain.page), np.asarray(filmed.page)), (
        "augmentation must change the page"
    )
    # the returned cache re-uses the SAME inks + layout (nothing re-rendered) and keeps the page's seed;
    # it's a copy, not the original — recompose threads a re-rolled seed through it when one is given
    assert plain.render is not None
    assert plain.render.inks is cache.inks and plain.render.layout is cache.layout
    assert plain.render.seed == cache.seed


def test_live_update_reseats_on_spacing_change_but_falls_back_on_rewrap():
    from diffu_studio.page import live_update_page

    cache = _placeholder_cache(width=640)

    # bigger line gap only re-positions the same lines → re-seat the cached inks (no model)
    wider = cache.gen.model_copy(update={"line_gap": 0.6})
    step = live_update_page(cache, wider)
    assert step.phase == "final" and step.page is not None
    assert step.page_labels is not None and len(step.page_labels.lines) == len(cache.layout.boxes)

    # width is a GENERATE-time knob now: live update keeps the cached spec (only the gaps swap), so a
    # width change neither re-wraps nor resizes — the cached geometry survives the spacing tweak
    narrow = cache.gen.model_copy(update={"width": 260})
    kept = live_update_page(cache, narrow)
    assert kept.phase == "final" and kept.page is not None
    assert kept.page.width == cache.layout.spec.width, "cached spec must win over the request's width"

    # the safety net is still there: text that can't map onto the cached inks → error, never a mis-labelled page
    reworded = cache.model_copy(update={"paragraphs": ["totally different words that wrap another way"]})
    fallback = live_update_page(reworded, wider)
    assert fallback.phase == "error" and fallback.page is None and "re-wrap" in fallback.status


def test_engine_caches_and_evicts_oldest(monkeypatch):
    calls: list[str] = []

    def _fake_load(path: str, _arch, _device, *, lora=None, lora_scale=1.0):
        calls.append(path)
        return object()  # a stand-in LoadedModel; the engine only stores + counts it

    monkeypatch.setattr(engine_mod, "load_model", _fake_load)
    engine = StudioEngine(device=resolve_device("cpu"), max_cached_models=2)

    a1 = engine.model("a")
    assert engine.model("a") is a1  # cached: no reload
    engine.model("b")
    engine.model("c")  # evicts "a" (oldest)
    engine.model("a")  # reloaded
    assert calls == ["a", "b", "c", "a"]


def test_engine_cache_key_includes_lora(monkeypatch):
    # The same base checkpoint with a different adapter (or scale) must be a distinct cache entry, so
    # switching LoRA reloads instead of returning the stale base model.
    seen: list[tuple[str, str | None, float]] = []

    def _fake_load(path: str, _arch, _device, *, lora=None, lora_scale=1.0):
        seen.append((path, lora, lora_scale))
        return object()

    monkeypatch.setattr(engine_mod, "load_model", _fake_load)
    engine = StudioEngine(device=resolve_device("cpu"), max_cached_models=8)

    base = engine.model("ckpt")
    assert engine.model("ckpt") is base  # no lora, cached
    with_lora = engine.model("ckpt", lora="runs/iam_lora/final")
    assert with_lora is not base  # same base, adapter added → reload
    assert engine.model("ckpt", lora="runs/iam_lora/final") is with_lora  # same lora → cached
    scaled = engine.model("ckpt", lora="runs/iam_lora/final", lora_scale=0.5)
    assert scaled is not with_lora  # same adapter, different scale → reload
    assert seen == [
        ("ckpt", None, 1.0),
        ("ckpt", "runs/iam_lora/final", 1.0),
        ("ckpt", "runs/iam_lora/final", 0.5),
    ]
