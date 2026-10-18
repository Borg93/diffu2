"""Tests for the PiD upscaler's CPU fallback seam — without the ``pid`` install, ``pid_upscale`` is a
logged no-op that hands the caller's page back unchanged (the contract PidActor and the studio rely on)."""

from __future__ import annotations

import importlib.util
import sys

from diffu_page import upscale
from PIL import Image


def test_pid_upscale_returns_the_input_unchanged_when_pid_is_unavailable(monkeypatch):
    monkeypatch.setattr(upscale, "pid_available", lambda: False)
    img = Image.new("RGB", (48, 32), "white")
    assert upscale.pid_upscale(img) is img, "the fallback must hand back the very same image object"


def test_pid_upscale_passes_none_through(monkeypatch):
    monkeypatch.setattr(upscale, "pid_available", lambda: False)
    assert upscale.pid_upscale(None) is None


def test_pid_available_is_import_probe_only():
    # the probe must agree with the install (present when the [upscale] extra is synced, absent when
    # not) WITHOUT importing pid — importing would drag in torch/CUDA at probe time
    assert upscale.pid_available() is (importlib.util.find_spec("pid") is not None)
    assert "pid" not in sys.modules, "pid_available must probe, not import"
