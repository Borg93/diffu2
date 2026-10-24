"""A bucket sampler's batches interleaved across buckets, resumable mid-epoch. No NeMo import, so it runs and tests off
the NeMo environment; ``diffu_train.nemo`` puts it around NeMo's ``SequentialBucketSampler``."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, override

import torch
from torch.utils.data import Sampler


class BucketBatches(Protocol):
    """What interleaving needs of a bucket sampler (NeMo's ``SequentialBucketSampler``): its batches, each of one crop
    shape, in an order its seed and epoch decide."""

    seed: int

    def __iter__(self) -> Iterator[list[int]]: ...

    def __len__(self) -> int: ...

    def set_epoch(self, epoch: int) -> None: ...


class InterleavedBucketSampler(Sampler[list[int]]):
    """The batches of a bucket sampler, shuffled across buckets; resumable mid-epoch."""

    def __init__(self, buckets: BucketBatches) -> None:
        super().__init__()
        self._buckets = buckets
        self._epoch = 0
        self._resume_at = 0
        self._yielded = 0

    def __len__(self) -> int:
        return len(self._buckets)

    @override
    def __iter__(self) -> Iterator[list[int]]:
        batches = list(self._buckets)
        order = torch.randperm(len(batches), generator=torch.Generator().manual_seed(self._buckets.seed + self._epoch))
        self._yielded, self._resume_at = self._resume_at, 0
        for index in order[self._yielded :].tolist():
            self._yielded += 1
            yield batches[index]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._buckets.set_epoch(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch, "batches_yielded": self._yielded}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.set_epoch(state["epoch"])
        self._resume_at = state["batches_yielded"]
