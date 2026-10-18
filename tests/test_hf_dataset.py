"""Column auto-detection for the HF dataset adapter (pure logic — no `datasets` needed)."""

from __future__ import annotations

import pytest

from diffu.data.hf_dataset import _pick


def test_pick_exact_and_case_insensitive() -> None:
    assert _pick(["image", "text"], ["image", "img"]) == "image"
    assert _pick(["Text", "Image"], ["text"]) == "Text"  # case-insensitive fallback


def test_pick_prefers_first_candidate() -> None:
    assert _pick(["transcription", "text"], ["text", "transcription"]) == "text"


def test_pick_optional_returns_none() -> None:
    assert _pick(["image", "text"], ["writer", "author"], required=False) is None


def test_pick_required_raises() -> None:
    with pytest.raises(ValueError, match="could not find"):
        _pick(["image"], ["text", "transcription"])
