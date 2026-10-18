"""CER metric."""

from __future__ import annotations

from diffu.model.metrics import cer


def test_cer_identical_is_zero():
    assert cer(["Göteborg"], ["Göteborg"]) == 0.0


def test_cer_single_substitution():
    # one wrong char out of 8 -> 1/8
    assert cer(["Goteborg"], ["Göteborg"]) == 1 / 8


def test_cer_batch_is_length_weighted():
    # 1 edit over (2 + 4) reference chars
    assert cer(["xx", "abcd"], ["xy", "abcd"]) == 1 / 6


def test_cer_empty_prediction():
    assert cer([""], ["abc"]) == 1.0
