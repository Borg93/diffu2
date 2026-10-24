"""The interleaved bucket sampler off the NeMo environment, around a small bucket sampler of the same shape as NeMo's:
a run resumed mid-epoch from its state dict draws exactly the batches the uninterrupted epoch had left."""

from __future__ import annotations

import random
from collections.abc import Iterator
from itertools import islice

from diffu_train.sampler import InterleavedBucketSampler


class _Buckets:
    """Three buckets of four one-crop batches, the batches of each bucket in an order of the seed and epoch, the
    buckets one after another, as NeMo's SequentialBucketSampler yields them."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        for bucket in range(3):
            yield from ([100 * bucket + k] for k in rng.sample(range(4), 4))

    def __len__(self) -> int:
        return 12

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


def _at_epoch(epoch: int) -> InterleavedBucketSampler:
    sampler = InterleavedBucketSampler(_Buckets(seed=3))
    sampler.set_epoch(epoch)
    return sampler


def test_interleaved_sampler_resumed_mid_epoch_draws_exactly_the_rest_of_the_epoch() -> None:
    whole = list(_at_epoch(2))
    stopped = _at_epoch(2)
    drawn = list(islice(iter(stopped), 5))
    resumed = InterleavedBucketSampler(_Buckets(seed=3))
    resumed.load_state_dict(stopped.state_dict())

    rest = list(resumed)

    assert (drawn, rest) == (whole[:5], whole[5:])
