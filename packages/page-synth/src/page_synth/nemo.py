"""Interleaved aspect-ratio buckets for NeMo AutoModel's FLUX.3 LoRA training.

NeMo's ``SequentialBucketSampler`` walks the buckets one after another: every batch of one crop shape, then
every batch of the next. A run shorter than an epoch (v2: 1,600 steps x 4 crops over 11,329 crops) therefore
never sees some buckets at all, and the low-learning-rate tail of the schedule falls on whichever buckets come
last. Crop shape follows page layout, so that decides which archives the LoRA learns. This sampler takes the
batches NeMo's sampler makes (shuffled, padded and split across ranks, one crop shape per batch) and shuffles
them across buckets, so any stretch of training sees every bucket in proportion.

Use it from a NeMo config with this folder on PYTHONPATH, for text-to-image or image-edit (reference) data:

    data:
      dataloader:
        _target_: page_synth.nemo.InterleavedTextToImageDataloaderConfig  # or InterleavedImageEditDataloaderConfig
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import override

import torch
from nemo_automodel.components.datasets.diffusion.collate_fns import TextToImageDataloaderConfig
from nemo_automodel.components.datasets.diffusion.image_edit_dataset import ImageEditDataloaderConfig
from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild
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


def interleaved(built: DiffusionDataloaderBuild) -> DiffusionDataloaderBuild:
    """``built`` with its batches shuffled across buckets: the same dataset, collate and worker settings.

    Raises:
        TypeError: If NeMo built something other than a ``SequentialBucketSampler``.
    """
    if not isinstance(built.sampler, SequentialBucketSampler):
        raise TypeError(f"expected NeMo's SequentialBucketSampler, got {type(built.sampler).__name__}")
    loader = built.dataloader
    sampler = InterleavedBucketSampler(built.sampler)
    dataloader = StatefulDataLoader(
        loader.dataset,
        batch_sampler=sampler,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        generator=loader.generator,
    )
    return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler, drop_last=built.drop_last)


def _require_shuffle(shuffle: bool) -> None:
    if not shuffle:
        raise ValueError("interleaving the buckets shuffles them; use NeMo's config for a fixed order")


# Dataclasses, not Pydantic models: NeMo accepts a dataloader _target_ outside its own builders only as a
# dataclass with a build() (recipes/_typed_config.py, resolve_diffusion_dataloader).
@dataclass
class InterleavedTextToImageDataloaderConfig(TextToImageDataloaderConfig):
    """NeMo's text-to-image dataloader config, with the aspect buckets interleaved.

    Raises:
        ValueError: If ``shuffle`` is off: interleaving is a shuffle.
    """

    def __post_init__(self) -> None:
        _require_shuffle(self.shuffle)

    @override
    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        return interleaved(super().build(dp_rank=dp_rank, dp_world_size=dp_world_size, batch_size=batch_size))


@dataclass
class InterleavedImageEditDataloaderConfig(ImageEditDataloaderConfig):
    """NeMo's image-edit (target + reference image) dataloader config, with the buckets interleaved.

    Raises:
        ValueError: If ``shuffle`` is off: interleaving is a shuffle.
    """

    def __post_init__(self) -> None:
        _require_shuffle(self.shuffle)

    @override
    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        return interleaved(super().build(dp_rank=dp_rank, dp_world_size=dp_world_size, batch_size=batch_size))
