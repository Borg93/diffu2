"""Our pages' latent stats, and a cache's latents handed to training as a repo with other stats normalizes them.

Qwen-Image's VAE normalizes its latents with stats measured on photos and video. Our pages (flat paper, thin strokes)
come out of that normalization with about a quarter of unit standard deviation and channel means off zero (TODO.md,
diffu), so flow matching's unit noise drowns a page long before its timestep schedule assumes. :func:`page_latent_stats`
measures the raw latents' mean and standard deviation per VAE channel over a cache's crops (a cache's latents are
normalized with its model's stats: undoing that gives the raw ones), for ``diffu-init --latent-stats``.
:class:`RenormalizedLatents` hands NeMo a cache's latents as a repo with other stats normalizes them, so the cache trains
that repo without being made again (``latents_of`` in ``diffu_train.nemo``). No NeMo import: it runs and tests anywhere.

    python -m diffu_train.latent_stats --cache data_flux3/cache_diffu_e2b_glyph --out runs/diffu/pages_latent_stats.json
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from rich.console import Console
from torch.utils.data import Dataset

from diffu.latents import LatentStats, renormalized, vae_stats
from diffu_train.cache import cache_model, cached_files
from diffu_train.dropout import Item, SampleSource

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()


def _own_latents(sample: Mapping[str, Any]) -> torch.Tensor:
    """A cached item's own latents: an image-edit pair's target (not its reference), a text-to-image crop's latent."""
    return sample["target_latent"] if "target_latent" in sample else sample["latent"]


def page_latent_stats(cache: Path) -> tuple[LatentStats, int]:
    """The mean and standard deviation per VAE channel of the raw latents of every crop in the finished cache at
    ``cache``, and how many crops that is.

    Raises:
        ValueError: If the cache is not finished, or names no diffu repo.
    """
    cached = vae_stats(cache_model(cache))
    channels = len(cached.latents_mean)
    raw_scale = LatentStats(latents_mean=(0.0,) * channels, latents_std=(1.0,) * channels)
    total = torch.zeros(channels, dtype=torch.float64)
    squares = torch.zeros(channels, dtype=torch.float64)
    values = 0
    files = cached_files(cache)
    for path in files:
        sample = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        raw = renormalized(_own_latents(sample).double(), cached, raw_scale).reshape(channels, -1)
        total += raw.sum(dim=1)
        squares += raw.square().sum(dim=1)
        values += raw.shape[1]
    mean = total / values
    std = (squares / values - mean.square()).sqrt()
    return LatentStats(latents_mean=tuple(mean.tolist()), latents_std=tuple(std.tolist())), len(files)


class RenormalizedLatents(Dataset[Item]):
    """A NeMo image-edit dataset whose target and reference latents, cached normalized with ``cached``, come out
    normalized with ``wanted``: what the cache would hold had a repo with those stats made it."""

    def __init__(self, dataset: SampleSource, *, cached: LatentStats, wanted: LatentStats) -> None:
        super().__init__()
        self._dataset = dataset
        self._cached = cached
        self._wanted = wanted

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Item:
        sample = self._dataset[index]
        return {
            **sample,
            "target_latent": renormalized(sample["target_latent"], self._cached, self._wanted),
            "context_latents": [
                renormalized(latents, self._cached, self._wanted) for latents in sample["context_latents"]
            ],
        }


@app.command()
def main(
    cache: Annotated[Path, typer.Option(help="a finished diffu cache (diffu_train.cache_diffu or cache_glyphs)")],
    out: Annotated[Path, typer.Option(help="the stats' JSON file, for diffu-init --latent-stats")],
) -> None:
    """Measure our pages' latent mean and standard deviation per VAE channel over every crop of a cache."""
    try:
        stats, crops = page_latent_stats(cache)
        cached = vae_stats(cache_model(cache))
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--cache") from error
    out.write_text(stats.model_dump_json(indent=1), encoding="utf-8")
    spread = [ours / theirs for ours, theirs in zip(stats.latents_std, cached.latents_std, strict=True)]
    console.print(
        f"{crops} crops -> {out}; normalized as cached, their channels' std is {min(spread):.3f}-{max(spread):.3f}"
    )


if __name__ == "__main__":
    app()
