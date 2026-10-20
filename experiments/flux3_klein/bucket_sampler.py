"""Interleaved aspect-ratio buckets for NeMo AutoModel's text-to-image LoRA training.

NeMo's ``SequentialBucketSampler`` walks the buckets one after another: every batch of one crop shape, then
every batch of the next. A run shorter than an epoch (v2: 1,600 steps x 4 crops over 11,329 crops) therefore
never sees some buckets at all, and the low-learning-rate tail of the schedule falls on whichever buckets come
last. Crop shape follows page layout, so that decides which archives the LoRA learns. This sampler takes the
batches NeMo's sampler makes (shuffled, padded and split across ranks, one crop shape per batch) and shuffles
them across buckets, so any stretch of training sees every bucket in proportion.

Use it from a NeMo config with this folder on PYTHONPATH:

    data:
      dataloader:
        _target_: bucket_sampler.build_interleaved_text_to_image_dataloader
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, override

import torch
from nemo_automodel.components.datasets.diffusion.collate_fns import TextToImageDataloaderConfig
from nemo_automodel.components.datasets.diffusion.sampler import SequentialBucketSampler
from torch.utils.data import Sampler
from torchdata.stateful_dataloader import StatefulDataLoader


class InterleavedBucketSampler(Sampler[list[int]]):
    """The batches of a ``SequentialBucketSampler``, shuffled across buckets; resumable mid-epoch."""

    def __init__(self, buckets: SequentialBucketSampler) -> None:
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
        order = torch.randperm(
            len(batches), generator=torch.Generator().manual_seed(self._buckets.seed + self._epoch)
        )
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


def build_interleaved_text_to_image_dataloader(
    *, batch_size: int = 1, dp_rank: int = 0, dp_world_size: int = 1, **config: Any
) -> tuple[StatefulDataLoader, InterleavedBucketSampler]:
    """NeMo's ``build_text_to_image_multiresolution_dataloader`` with interleaved buckets.

    ``config`` holds the dataloader section of the training config (``cache_dir``, ``num_workers``, ...), the
    fields of NeMo's ``TextToImageDataloaderConfig``.

    Raises:
        ValueError: If ``shuffle`` is off: interleaving is a shuffle.
    """
    settings = TextToImageDataloaderConfig(**config)
    if not settings.shuffle:
        raise ValueError("interleaving the buckets shuffles them; use NeMo's builder for a fixed order")
    built = settings.build(batch_size=batch_size, dp_rank=dp_rank, dp_world_size=dp_world_size)
    sampler = InterleavedBucketSampler(built.sampler)
    dataloader = StatefulDataLoader(
        built.dataloader.dataset,
        batch_sampler=sampler,
        collate_fn=built.dataloader.collate_fn,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        prefetch_factor=settings.prefetch_factor if settings.num_workers > 0 else None,
        persistent_workers=settings.num_workers > 0,
    )
    return dataloader, sampler
