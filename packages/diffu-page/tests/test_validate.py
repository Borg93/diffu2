"""Validation metrics: IoU, CER, box conversion, and matching against ground truth."""

import pytest
from diffu_page.validate import MatchResult, char_error_rate, iou, match, poly_to_box


def test_iou() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_char_error_rate() -> None:
    assert char_error_rate("abc", "abc") == 0.0
    assert char_error_rate("abx", "abc") == pytest.approx(1 / 3)
    assert char_error_rate("", "abc") == pytest.approx(1.0)


def test_poly_to_box() -> None:
    assert poly_to_box([(5, 7), (15, 7), (15, 20), (5, 20)]) == (5, 7, 15, 20)


def test_match_perfect() -> None:
    result = match([((0, 0, 10, 10), "hej")], [((0, 0, 10, 10), "hej")])
    assert isinstance(result, MatchResult)
    assert result.matched == 1
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.mean_cer == 0.0
