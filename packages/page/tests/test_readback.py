"""The character error rate every drawn line's read-back is scored with."""

from __future__ import annotations

import pytest

from diffu_page.readback import cer


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
