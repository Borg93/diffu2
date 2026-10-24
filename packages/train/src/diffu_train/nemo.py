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
        _target_: diffu_train.nemo.InterleavedTextToImageDataloaderConfig  # or InterleavedImageEditDataloaderConfig

Both configs also take ``caption_dropout``: that share of the items trains on the empty prompt
(``diffu_train.dropout``), read from the ``empty_prompt.pt`` that ``diffu_train.cache_pairs`` (FLUX.3's pairs) or
``diffu_train.cache_diffu`` (diffu's crops) writes into the cache.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass  # noqa: TID251 -- NeMo builds dataloaders only from dataclass configs
from functools import partial
from pathlib import Path
from typing import override

from nemo_automodel.components.datasets.diffusion.collate_fns import TextToImageDataloaderConfig
from nemo_automodel.components.datasets.diffusion.image_edit_dataset import ImageEditDataloaderConfig
from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild
from nemo_automodel.components.datasets.diffusion.sampler import SequentialBucketSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from diffu_train.dropout import EMPTY_PROMPT_FILE, CaptionDropout, EmptyPrompt, Item, Sample, SampleSource
from diffu_train.sampler import InterleavedBucketSampler

type DatasetWrap = Callable[[SampleSource], SampleSource]

logger = logging.getLogger(__name__)


def _as_built(dataset: SampleSource) -> SampleSource:
    return dataset


def interleaved(built: DiffusionDataloaderBuild, wrap: DatasetWrap = _as_built) -> DiffusionDataloaderBuild:
    """``built`` with its batches shuffled across buckets: the same collate and worker settings, its dataset passed
    through ``wrap`` (the buckets stay those of the dataset NeMo built).

    Raises:
        TypeError: If NeMo built something other than a ``SequentialBucketSampler``.
    """
    if not isinstance(built.sampler, SequentialBucketSampler):
        raise TypeError(f"expected NeMo's SequentialBucketSampler, got {type(built.sampler).__name__}")
    loader = built.dataloader
    sampler = InterleavedBucketSampler(built.sampler)
    dataloader = StatefulDataLoader(
        wrap(loader.dataset),
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


def _require_share(caption_dropout: float) -> None:
    if not 0.0 <= caption_dropout < 1.0:
        raise ValueError(f"caption_dropout is a share of the items in [0, 1), got {caption_dropout}")


def _with_dropout(
    built: DiffusionDataloaderBuild, *, cache_dir: str, share: float, blank: Callable[[EmptyPrompt, Sample], Item]
) -> DiffusionDataloaderBuild:
    """``built`` interleaved, ``share`` of its items on the cache's empty prompt, put in by ``blank`` (the
    ``EmptyPrompt`` method for an item of this kind)."""
    if not share:
        return interleaved(built)
    cache = Path(cache_dir)
    empty = EmptyPrompt.load(cache)
    logger.info("caption dropout: %s of the items on the empty prompt in %s", share, cache / EMPTY_PROMPT_FILE)
    return interleaved(built, lambda dataset: CaptionDropout(dataset, partial(blank, empty), share=share))


# Dataclasses, not Pydantic models: NeMo accepts a dataloader _target_ outside its own builders only as a
# dataclass with a build() (recipes/_typed_config.py, resolve_diffusion_dataloader).
@dataclass
class InterleavedTextToImageDataloaderConfig(TextToImageDataloaderConfig):
    """NeMo's text-to-image dataloader config, with the aspect buckets interleaved and ``caption_dropout`` of the crops
    trained on the empty prompt (``diffu_train.dropout``; 0 trains every caption).

    Raises:
        ValueError: If ``shuffle`` is off (interleaving is a shuffle), or ``caption_dropout`` is not in [0, 1).
    """

    caption_dropout: float = 0.0

    def __post_init__(self) -> None:
        _require_shuffle(self.shuffle)
        _require_share(self.caption_dropout)

    @override
    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        built = super().build(dp_rank=dp_rank, dp_world_size=dp_world_size, batch_size=batch_size)
        return _with_dropout(
            built, cache_dir=self.cache_dir, share=self.caption_dropout, blank=EmptyPrompt.into_text_to_image
        )


@dataclass
class InterleavedImageEditDataloaderConfig(ImageEditDataloaderConfig):
    """NeMo's image-edit (target + reference image) dataloader config, with the buckets interleaved and
    ``caption_dropout`` of the pairs trained on the empty prompt (``diffu_train.dropout``; 0 trains every caption).

    Raises:
        ValueError: If ``shuffle`` is off (interleaving is a shuffle), or ``caption_dropout`` is not in [0, 1).
    """

    caption_dropout: float = 0.0

    def __post_init__(self) -> None:
        _require_shuffle(self.shuffle)
        _require_share(self.caption_dropout)

    @override
    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        built = super().build(dp_rank=dp_rank, dp_world_size=dp_world_size, batch_size=batch_size)
        return _with_dropout(
            built, cache_dir=self.cache_dir, share=self.caption_dropout, blank=EmptyPrompt.into_image_edit
        )
