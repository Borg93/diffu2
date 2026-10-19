"""Tests for DiffuActor's width-bucketing — the pure plan and the batch-order reassembly (no Ray, no GPU).

``bucket_plan`` is pure arithmetic. The ``__call__`` tests build a DiffuActor via ``__new__`` with the
attrs the real ctor would set (the ctor itself needs a safetensors checkpoint + CUDA) and monkeypatch the
``diffu.generate`` helpers it imports at call time — the fake model tags each output with its input text
and canvas width, so misaligned reassembly is directly visible.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffu_runner.actors import DiffuActor, bucket_plan

# ---------------------------------------------------------------- bucket_plan (pure)


def test_plan_groups_equal_buckets_ascending() -> None:
    # naturals 100/90 share the 128 bucket; 130 rounds up past it; plan is ascending by width
    plan = bucket_plan([100, 500, 90, 1000, 130], bucket_px=128, max_width=1024)
    assert plan == [(128, [0, 2]), (256, [4]), (512, [1]), (1024, [3])]


def test_plan_covers_every_index_exactly_once() -> None:
    widths = [17, 1024, 3, 640, 640, 999, 128, 1]
    plan = bucket_plan(widths, bucket_px=128, max_width=1024)
    flat = [i for _, idxs in plan for i in idxs]
    assert sorted(flat) == list(range(len(widths))), "a lost/duplicated index mislabels training rows"
    assert [w for w, _ in plan] == sorted({w for w, _ in plan}), "buckets must come out ascending"


def test_plan_clamps_to_bucket_floor_and_max_width() -> None:
    # 0 (degenerate) never drops below one bucket; a huge natural width caps at max_width
    assert bucket_plan([0, 50_000], bucket_px=128, max_width=1024) == [(128, [0]), (1024, [1])]


def test_plan_keeps_16px_canvas_invariant() -> None:
    # a bucket_px that is NOT a multiple of 16 must still yield 16px-multiple canvases (the ctor's
    # ((w+15)//16)*16 invariant — the bucket must stay divisible by the VAE downscale factor)
    plan = bucket_plan([1, 101, 999], bucket_px=100, max_width=1000)
    assert [w for w, _ in plan] == [112, 208, 1008]
    assert all(w % 16 == 0 for w, _ in plan)


def test_plan_rejects_nonpositive_bucket() -> None:
    with pytest.raises(ValueError, match="bucket_px"):
        bucket_plan([100], bucket_px=0, max_width=1024)


# ---------------------------------------------------------------- __call__ reassembly (fake model)


class _FakeModel:
    """Tags each output with (text, latent width) so the test can check per-row alignment + bucketing."""

    def generate(self, texts, style, latent_hw, num_steps, cfg_scale):  # noqa: ANN001 — mirrors Diffu.generate
        assert style.shape[0] == len(texts), "style rows must be sliced with the same bucket indices"
        return [f"{t}|lw={latent_hw[1]}" for t in texts]


def _stub_actor() -> DiffuActor:
    """A DiffuActor with the attrs the real ctor would set, minus torch weights / CUDA."""
    actor = DiffuActor.__new__(DiffuActor)
    actor._model = _FakeModel()
    cfg_stub: Any = SimpleNamespace(  # Any: a duck-typed stand-in for the ctor's diffu Config
        data=SimpleNamespace(line_height=128), vae=SimpleNamespace(downscale_factor=8)
    )
    actor._cfg = cfg_stub
    actor._steps, actor._scale = 4, 5.0
    actor._latent_h = 16
    actor._max_width = 1024
    actor._bucket_px = 128
    actor._device = "cpu"
    actor._dtype = torch.float32
    return actor


@pytest.fixture
def fake_generate_env(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the diffu.generate helpers __call__ imports; returns the texts auto_canvas_width saw."""
    measured: list[str] = []

    def fake_auto_canvas_width(model, text):  # noqa: ANN001
        measured.append(text)
        return 16 * len(text)  # deterministic content-proportional width

    monkeypatch.setattr("diffu.generate.auto_canvas_width", fake_auto_canvas_width)
    monkeypatch.setattr("diffu.generate.load_style", lambda p: torch.zeros(3, 4, 4))
    monkeypatch.setattr("diffu.generate.to_pil", lambda img: img)  # fake generate yields tag strings
    monkeypatch.setattr("diffu_runner.actors._png_bytes", lambda img: img.encode())
    return measured


def test_call_reassembles_outputs_in_input_order(fake_generate_env: list[str]) -> None:
    texts = ["a" * 4, "b" * 40, "c" * 6, "d" * 64, "e" * 12]  # buckets: 128, 640, 128, 1024, 256
    batch = {"text": texts, "style_path": [f"s{i}.png" for i in range(5)]}
    out = _stub_actor()(batch)

    # output[i] must correspond to texts[i] even though buckets scrambled the generate order
    tags = [p.decode() for p in out["png"]]
    assert [t.split("|")[0] for t in tags] == texts
    # and the canvas actually varied per bucket: 128/8=16, 640/8=80, 1024/8=128, 256/8=32
    assert [t.split("lw=")[1] for t in tags] == ["16", "80", "16", "128", "32"]


def test_call_passes_every_input_column_through(fake_generate_env: list[str]) -> None:
    batch = {
        "text": ["ab", "cdefgh" * 8],
        "style_path": ["s0.png", "s1.png"],
        "doc_id": ["doc0", "doc1"],  # upcoming metadata columns must survive to the gate/write stages
        "line_idx": [0, 1],
    }
    out = _stub_actor()(batch)
    for col, values in batch.items():
        assert out[col] == list(values), f"input column {col!r} was dropped or reordered"
    assert len(out["png"]) == 2


def test_call_measures_empty_text_as_single_space(fake_generate_env: list[str]) -> None:
    # the actor must measure a blank row as " ", not "" (the page path's convention)
    _stub_actor()({"text": ["", "hello"], "style_path": ["s.png", "s.png"]})
    assert fake_generate_env[0] == " "
