"""Tests for the batched renderer's width-bucketing plan — pure arithmetic, no torch/model needed."""

from __future__ import annotations

import pytest
from diffu_page.render.diffu import bucket_render_batches


def test_widths_quantise_up_and_equal_buckets_group_together():
    batches = bucket_render_batches([1, 16, 17, 30], bucket_px=16, max_batch=8)
    assert batches == [(16, [0, 1]), (32, [2, 3])]


def test_groups_split_into_chunks_of_max_batch():
    batches = bucket_render_batches([10, 10, 10, 10, 10], bucket_px=16, max_batch=2)
    assert batches == [(16, [0, 1]), (16, [2, 3]), (16, [4])]


def test_every_job_is_planned_exactly_once_in_ascending_bucket_order():
    widths = [130, 3, 64, 65, 17, 1024, 48, 47]
    batches = bucket_render_batches(widths, bucket_px=16, max_batch=3)
    planned = [k for _, chunk in batches for k in chunk]
    assert sorted(planned) == list(range(len(widths))), "each job rendered exactly once"
    assert [w for w, _ in batches] == sorted(w for w, _ in batches), "ascending bucket widths"
    for width, chunk in batches:
        assert width % 16 == 0
        assert all(widths[k] <= width for k in chunk), "a bucket never truncates its jobs"
        assert len(chunk) <= 3


def test_no_jobs_yields_no_batches():
    assert bucket_render_batches([], bucket_px=16, max_batch=4) == []


@pytest.mark.parametrize(("bucket_px", "max_batch"), [(0, 4), (16, 0), (-1, -1)])
def test_degenerate_parameters_fail_fast(bucket_px, max_batch):
    with pytest.raises(ValueError, match="must be >= 1"):
        bucket_render_batches([10], bucket_px=bucket_px, max_batch=max_batch)
