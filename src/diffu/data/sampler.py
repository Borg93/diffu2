"""Width-bucketing batch sampler for variable-width handwriting lines.

Lines share a fixed height but vary widely in width (GT-Lejonet: p50 ~560, max ~2500 px). With a plain
shuffle sampler each batch is padded to its OWN max width, so a batch mixing a 2500 px line with several
200 px lines wastes most of its VAE+DiT compute on blank padding — measured at **~54.5%** of the
width-FLOPs at batch 32. It also makes the latent width (and token count) differ almost every batch,
which makes ``torch.compile`` recompile per shape.

This sampler groups lines of SIMILAR width into each batch ("length bucketing"), cutting padding waste
to ~1% and collapsing the hundreds of distinct shapes into a handful of buckets (so ``torch.compile``
stops thrashing). Strategy: shuffle globally, cut into megabatches, sort each megabatch by width, cut
into batches, shuffle batch order — width-homogeneous batches whose composition still varies per epoch.

DDP: this yields the GLOBAL batch list; under ``accelerate`` the prepared ``DataLoader`` wraps it in a
``BatchSamplerShard`` that hands each rank every Nth batch (disjoint, equal counts with ``drop_last``).
All ranks must therefore produce the SAME global order — which they do, because the shuffle is seeded by
``(seed, epoch)`` and every rank advances the epoch counter in lockstep. No cross-rank shape sync is
needed: DDP all-reduces gradients (param-shaped, independent of input width), so ranks may hold
different-width batches on the same step.

Caveat: bucketing makes batches width-correlated rather than i.i.d. — a small but real training-dynamics
change (not bit-identical like the other perf wins). Opt-in (``--bucket-width``); validate gen_CER before
making it the default.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator

from PIL import Image
from torch.utils.data import Sampler


def line_widths(
    rows: list[dict[str, str]], *, max_width: int | None = None, cache_path: str | None = None
) -> list[int]:
    """Per-row line width in px, capped at ``max_width`` (``None`` = the raw widths).

    Crops are stored at a fixed height, so the on-disk image width IS the line width (what the collate
    pads to). Reading ~1.6M PNG headers is slow, so the RAW widths are cached to a ``cache_path`` sidecar
    and reused on later runs; the cap is applied on read, so a changed ``max_width`` never reuses stale
    capped widths. Cache key = format + row count + first/last image path: a re-prepped manifest with
    the SAME length but different crops must not silently reuse stale widths.
    """
    fingerprint = {
        "format": 2,  # 2: raw widths (format 1 stored them capped, without the cap in the key)
        "n": len(rows),
        "first": rows[0]["image"] if rows else "",
        "last": rows[-1]["image"] if rows else "",
    }
    widths: list[int] | None = None
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            if all(cached.get(k) == v for k, v in fingerprint.items()):
                widths = cached["widths"]
        except (OSError, ValueError, KeyError):
            pass  # stale/corrupt cache -> recompute
    if widths is None:
        widths = []
        for row in rows:
            with Image.open(row["image"]) as im:
                widths.append(im.width)
        if cache_path:
            with open(cache_path, "w") as fh:
                json.dump({**fingerprint, "widths": widths}, fh)
    return widths if max_width is None else [min(w, max_width) for w in widths]


class WidthBucketBatchSampler(Sampler[list[int]]):
    """Yield batches of dataset indices whose lines have SIMILAR width (see module docstring).

    Each epoch: shuffle all rows, cut them into megabatches of ``MEGABATCH × batch_size``, sort each by
    width, cut into full batches (a partial last batch is dropped — DDP needs equal batch counts per rank),
    then shuffle the batch order. A larger megabatch gives more width-homogeneous batches but less
    cross-batch randomness.

    Args:
        widths: Per-row line width, aligned to the dataset's rows.
        batch_size: Lines per batch (per rank — accelerate shards whole batches across ranks).
        seed: Base RNG seed; combined with the epoch so all ranks shuffle identically.
    """

    MEGABATCH = 20

    def __init__(self, widths: list[int], batch_size: int, *, seed: int = 0) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.widths = widths
        self.batch_size = batch_size
        self.seed = seed
        self.megabatch = self.MEGABATCH * batch_size
        self._epoch = 0
        n = len(widths)
        self._nbatches = sum(min(self.megabatch, n - s) // batch_size for s in range(0, n, self.megabatch))

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch. Single-process accelerate calls this; under DDP accelerate's ``BatchSamplerShard``
        doesn't forward it, and the per-``__iter__`` auto-advance keeps the ranks in lockstep instead."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._nbatches

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self._epoch
        idx = list(range(len(self.widths)))
        random.Random(self.seed + epoch).shuffle(idx)
        batches: list[list[int]] = []
        for s in range(0, len(idx), self.megabatch):
            mega = sorted(idx[s : s + self.megabatch], key=self.widths.__getitem__)
            batches += [
                mega[b : b + self.batch_size]
                for b in range(0, len(mega) - self.batch_size + 1, self.batch_size)
            ]
        random.Random(self.seed + epoch + 1_000_003).shuffle(batches)  # decorrelated sub-seed
        self._epoch += 1  # next epoch reshuffles; lockstep across ranks (same seed, same call count)
        yield from batches
