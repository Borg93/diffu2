"""The character error rate every drawn line's read-back is scored with, and the numbers it must read exactly."""

from __future__ import annotations

import pytest

from diffu_page.readback import cer, numbers_differ


@pytest.mark.parametrize(
    ("pred", "ref", "rate"),
    [
        pytest.param("maj", "maj", 0.0, id="same"),
        pytest.param("mej", "maj", 1 / 3, id="substitution"),
        pytest.param("mj", "maj", 1 / 3, id="deletion"),
        pytest.param("majj", "maj", 1 / 3, id="insertion"),
        pytest.param("", "maj", 1.0, id="nothing-read"),
        pytest.param("a", "", 1.0, id="empty-reference"),
    ],
)
def test_cer_counts_each_edit_per_reference_character(pred: str, ref: str, rate: float) -> None:
    assert cer(pred, ref) == rate


@pytest.mark.parametrize(
    ("pred", "ref", "differ"),
    [
        pytest.param("12 Junii 1826.", "12 Junii 1826.", False, id="same"),
        pytest.param("Stockholm 12 Iunij 1826", "12 Junii 1826.", False, id="letters-only"),
        pytest.param("10 Junii 1826.", "12 Junii 1826.", True, id="one-digit"),
        pytest.param("12 Junii", "12 Junii 1826.", True, id="number-missing"),
        pytest.param("1826 12 Junii", "12 Junii 1826.", True, id="order"),
        pytest.param("Junii", "Junii", False, id="no-numbers"),
    ],
)
def test_numbers_differ_only_when_a_number_reads_otherwise(pred: str, ref: str, differ: bool) -> None:
    assert numbers_differ(pred, ref) is differ
