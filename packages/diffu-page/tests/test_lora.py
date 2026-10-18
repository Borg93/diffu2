"""LoRA plumbing: the ``lora``/``lora_scale`` knobs reach both Diffu renderers — no model/torch needed.

Verifies the wiring only (config field → ``build_renderer`` → renderer attribute); the actual adapter
load happens lazily in ``_ensure`` on first render, so these stay CPU-only (``_model is None``).
"""

from __future__ import annotations

from pathlib import Path

from diffu_page.config import GenConfig, Renderer
from diffu_page.generation import build_renderer
from diffu_page.render.diffu import BatchedDiffuRenderer, DiffuRenderer


def _diffu_cfg(**over: object) -> GenConfig:
    base = {
        "renderer": Renderer.diffu,
        "ckpt": Path("base.safetensors"),
        "style": Path("ref_line.png"),
        "lora": "runs/iam_lora/final",
        "lora_scale": 0.7,
    }
    return GenConfig(**{**base, **over})


def test_gen_config_carries_lora_knobs():
    cfg = _diffu_cfg()
    assert cfg.lora == "runs/iam_lora/final"
    assert cfg.lora_scale == 0.7


def test_gen_config_lora_defaults_to_none_full_scale():
    cfg = GenConfig()
    assert cfg.lora is None
    assert cfg.lora_scale == 1.0


def test_build_renderer_threads_lora_to_per_line_renderer():
    r = build_renderer(_diffu_cfg(fast=False))
    assert isinstance(r, DiffuRenderer)
    assert r.lora == "runs/iam_lora/final" and r.lora_scale == 0.7
    assert r._model is None  # lazy: adapter loads in _ensure on first render, not at construction


def test_build_renderer_threads_lora_to_batched_renderer():
    r = build_renderer(_diffu_cfg(fast=True))
    assert isinstance(r, BatchedDiffuRenderer)
    assert r.lora == "runs/iam_lora/final" and r.lora_scale == 0.7
    assert r._model is None
