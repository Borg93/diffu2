"""Tests for the width-bucketing batch sampler (src/diffu/data/sampler.py)."""

from __future__ import annotations

import random

from diffu.data.sampler import WidthBucketBatchSampler


def _widths(n: int = 1000, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(20, 2500) for _ in range(n)]


def test_batches_are_width_homogeneous() -> None:
    widths = _widths()
    sampler = WidthBucketBatchSampler(widths, batch_size=32, seed=0)
    spreads = [max(widths[i] for i in b) - min(widths[i] for i in b) for b in sampler]
    global_spread = max(widths) - min(widths)
    assert max(spreads) < 0.25 * global_spread  # within-batch spread << global spread


def test_len_matches_iter() -> None:
    sampler = WidthBucketBatchSampler(_widths(), batch_size=32, seed=0)
    assert len(list(sampler)) == len(sampler)


def test_all_batches_full_with_drop_last() -> None:
    sampler = WidthBucketBatchSampler(_widths(1000), batch_size=32, seed=0)
    assert all(len(b) == 32 for b in sampler)


def test_indices_valid_and_unique_within_epoch() -> None:
    widths = _widths(1000)
    sampler = WidthBucketBatchSampler(widths, batch_size=32, seed=0)
    flat = [i for b in sampler for i in b]
    assert len(flat) == len(set(flat))  # no index appears twice in one epoch
    assert all(0 <= i < len(widths) for i in flat)


def test_deterministic_across_ranks() -> None:
    # Two samplers with the same seed must yield the SAME global batch order — the DDP requirement, since
    # accelerate's BatchSamplerShard splits this global list across ranks.
    widths = _widths()
    a = list(WidthBucketBatchSampler(widths, batch_size=32, seed=7))
    b = list(WidthBucketBatchSampler(widths, batch_size=32, seed=7))
    assert a == b


def test_reshuffles_across_epochs() -> None:
    sampler = WidthBucketBatchSampler(_widths(), batch_size=32, seed=0)
    assert list(sampler) != list(sampler)  # composition/order varies epoch to epoch


def test_line_widths_caches_raw_widths_and_caps_on_read(tmp_path):
    from PIL import Image

    from diffu.data.sampler import line_widths

    rows = []
    for i, w in enumerate((40, 300)):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (w, 8), "white").save(path)
        rows.append({"image": str(path)})
    cache = str(tmp_path / "train.widths.json")
    assert line_widths(rows, max_width=100, cache_path=cache) == [40, 100]
    assert line_widths(rows, max_width=500, cache_path=cache) == [40, 300]  # a new cap, not stale widths
    assert line_widths(rows, cache_path=cache) == [40, 300]
