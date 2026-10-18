"""Tests for the quality gate — the pure logic that keeps model misfires out of the dataset (no Ray/GPU)."""

from __future__ import annotations

import numpy as np
from diffu_runner.gate import GateConfig, cer_ok, ink_fraction, ink_ok
from PIL import Image


def _line(ink_frac: float, w: int = 400, h: int = 64) -> Image.Image:
    """A synthetic paper crop with a controlled fraction of dark (ink) pixels."""
    arr = np.full((h, w), 235, np.uint8)  # paper
    n_ink = int(ink_frac * w * h)
    idx = np.unravel_index(np.arange(n_ink), (h, w))
    arr[idx] = 20  # ink
    return Image.fromarray(arr)


def test_ink_fraction_tracks_darkness():
    assert ink_fraction(_line(0.0)) < 0.01
    assert 0.08 < ink_fraction(_line(0.10)) < 0.12


def test_ink_gate_rejects_blank_and_smear_keeps_normal():
    cfg = GateConfig()
    assert not ink_ok(_line(0.0), cfg), "a blank output is rejected"
    assert not ink_ok(_line(0.6), cfg), "a dense smear/blob is rejected"
    assert ink_ok(_line(0.12), cfg), "a normal line is kept"


def test_ink_gate_keeps_dense_but_legible_line():
    # a bold hand near the ink_max floor should survive the cheap gate (TrOCR then arbitrates)
    cfg = GateConfig()
    assert ink_ok(_line(0.40), cfg)


def test_cer_gate_threshold():
    cfg = GateConfig(cer_max=0.30)
    assert cer_ok(0.02, cfg) and cer_ok(0.30, cfg)
    assert not cer_ok(0.31, cfg)
