"""diffu-moss-cache-region-crops: a cache whose entries are given region crops in place of their whole sample, encoded once
on the GPU.

    diffu-moss-cache-region-crops --cache data_pages/cache_moss_pages --crops data_pages/region_crops_over/region_crops.jsonl \\
        --out data_pages/cache_moss_region_crops_over

A sample per text region (``diffu_page.draw.line_samples``' :func:`plan_region_crops`): each handwritten text region of
a target gets a crop of another text region of its own page (a window's: of its sibling window), cut at the region's
writing scale and placed on the DiT's position grid against the region's box, each crop a reference of its own at its
own offset (``MossTransformer``'s ``reference_offsets``). The crops are planned and cut on
the CPU by ``diffu_page.data.region_crop_pairs`` from a cache's entries (in their look change), which writes the manifest
read here (``region_crops.jsonl``, its build's record ``region_crops.json`` beside it, the crops' images under
``crops/``).

Each row of the manifest gives the cache entry it names its crops: written under ``--out`` by the same file name, its
target's latents and its prompt's embeddings (if the cache holds them) copied from the first cache as they are, its
whole sample left out, and each crop encoded by the VAE of the model directory the first cache was made with (or
``--model``'s) as the cache encodes a reference (:func:`diffu_moss.latents.encode_latents` of
:func:`diffu_moss.latents.image_pixels`), stored as ``region_crop_0``, ``region_crop_1``, ... in the row's order, in
bfloat16. The entry records its crops in that order (:class:`diffu_moss.train.cache.CachedRegionCrop`: their source and
target regions, sizes and offsets), and the cache's record what it was made from
(:class:`diffu_moss.train.records.RegionCropOrigin`, with the place its crops lie in, which a training on it is given
as ``--region-crops`` and records in its checkpoints' ``config.json``). Each file is written whole before it is named,
so a stopped run resumes; ``entries.jsonl`` and ``cache.json`` last. ``--shard i --shards n`` encodes only the rows at
the places ``i``, ``i + n``, ... and writes neither, as ``diffu-moss-cache-pairs`` shards a cache: a run without
``--shards`` over ``--out`` then finishes it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Self

import typer
from diffusers.modular_pipelines import ModularPipeline
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from diffu_moss.latents import encode_latents, image_pixels
from diffu_moss.train.cache import (
    CACHE_DTYPE,
    ENTRIES,
    CachedRegionCrop,
    CacheEntry,
    encoder_pipeline,
    read_entries,
    read_record,
    region_crop_tensor,
)
from diffu_moss.train.records import CACHE_RECORD, CacheRecord, JsonRecord, RegionCropOrigin
from diffu_moss.transformer import CropPlace

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

REGION_CROPS = "region_crops.jsonl"
REGION_CROP_BUILD = "region_crops.json"
CROP_IMAGES = "crops"
VAE = ("vae",)


class RegionCropFile(BaseModel):
    """One crop of a manifest row: its image (relative to the manifest's directory), the source region it is cut from and
    the target region whose box it styles (their blocks), its size (px, on the latent grid) and its centre's offset from the
    target's centre in latent rows and columns."""

    model_config = ConfigDict(frozen=True)

    file: str
    source: str
    target: str
    size: tuple[int, int]  # (width, height) px
    offset: tuple[float, float]  # (rows, columns)

    def cached(self) -> CachedRegionCrop:
        """The crop as a cache entry records it."""
        return CachedRegionCrop(source=self.source, target=self.target, size=self.size, offset=self.offset)


class RegionCropRow(BaseModel):
    """A cache entry's crops, in its target's region order: the entry by its file in the cache the manifest was built
    from, and its pair."""

    model_config = ConfigDict(frozen=True)

    entry: str
    pair: str
    crops: list[RegionCropFile] = Field(min_length=1)


class RegionCropBuild(JsonRecord):
    """``region_crops.json``: what a manifest of region crops was built from and how (the cache and the page-pair manifest
    it read, where its crops lie, at most how wide and tall, the seed that orders a source's regions, whether it
    served windows alone: left out of a build that served whole pages too), how many of the cache's entries it read and
    served, and the rest by why each was left out."""

    cache: str
    pairs: str
    place: CropPlace
    max_width: int
    max_height: int
    seed: int
    windows_only: bool = Field(default=False, exclude_if=lambda windows_only: not windows_only)
    entries_read: int
    served: int
    skipped: dict[str, int]


def read_region_crops(manifest: Path) -> list[RegionCropRow]:
    """The rows of a manifest of region crops."""
    with manifest.open(encoding="utf-8") as rows:
        return [RegionCropRow.model_validate_json(row) for row in rows if row.strip()]


class RegionCropCacheSettings(BaseModel):
    """Where a cache of region crops is encoded and which share of its rows this run encodes (``shard`` of ``shards``:
    the places ``shard``, ``shard + shards``, ...; one shard is every row)."""

    model_config = ConfigDict(frozen=True)

    device: str = "cuda"
    shard: int = Field(default=0, ge=0)
    shards: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _shard_within_shards(self) -> Self:
        if self.shard >= self.shards:
            raise ValueError(f"shard {self.shard} is not one of the {self.shards} shards (0 to {self.shards - 1})")
        return self


def crop_latents(pipe: ModularPipeline, path: Path, size: tuple[int, int]) -> Tensor:
    """The crop image at ``path`` as the DiT's latents, encoded as the cache encodes a reference.

    Raises:
        ValueError: If the image is not of ``size``, the size its row gives.
    """
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f"{path} is {image.size} px, its row gives {size}")
        return encode_latents(pipe.vae, image_pixels(image))


def region_crop_entry(
    pipe: ModularPipeline, cache: Path, entry: CacheEntry, row: RegionCropRow, crops: Path
) -> tuple[CacheEntry, dict[str, Tensor]]:
    """``entry`` of ``cache`` given ``row``'s crops (their images under ``crops``, the manifest's directory) in place
    of its sample: its entry and its tensors (in :data:`diffu_moss.train.cache.CACHE_DTYPE`, on the CPU), its target's
    latents and prompt embeddings those of ``cache`` as they are."""
    written = load_file(str(cache / entry.file))
    tensors = {name: written[name] for name in ("target", "prompt_embeds") if name in written}
    for index, crop in enumerate(row.crops):
        tensors[region_crop_tensor(index)] = crop_latents(pipe, crops / crop.file, crop.size)
    lined = entry.model_copy(update={"reference_size": None, "region_crops": [crop.cached() for crop in row.crops]})
    return lined, {name: tensor.to("cpu", CACHE_DTYPE).contiguous() for name, tensor in tensors.items()}


def write_region_crop_cache(
    cache: Path, crops: Path, out: Path, settings: RegionCropCacheSettings, *, model: Path | None = None
) -> CacheRecord | None:
    """The entries of the finished cache ``cache`` that the manifest of region crops ``crops`` gives crops, cached into
    ``out`` with them (the module's docstring) by the VAE of ``model`` (default: the model directory ``cache`` was
    made with), resuming the entries a stopped run wrote; with more than one shard, only ``settings.shard``'s.

    Returns:
        The cache's record, or None for a run of one shard of several (a run without shards over ``out`` finishes it).

    Raises:
        ValueError: If ``out`` holds a finished cache already, ``cache`` is not finished, the manifest was built from
            another cache or names an entry ``cache`` does not hold, or a crop's image is not the size its row gives.
        RuntimeError: If the VAE failed to load.
    """
    if (out / CACHE_RECORD).exists():
        raise ValueError(f"{out} holds a finished cache already; give a new directory")
    source = read_record(cache)
    build = RegionCropBuild.read(crops.parent / REGION_CROP_BUILD)
    if Path(build.cache).resolve() != cache.resolve():
        raise ValueError(f"{crops} was built from the cache {build.cache}, not {cache}")
    entries = {entry.file: entry for entry in read_entries(cache)}
    rows = read_region_crops(crops)
    unknown = [row.entry for row in rows if row.entry not in entries]
    if unknown:
        raise ValueError(f"{crops} names {len(unknown)} entries {cache} does not hold (the first: {unknown[0]})")
    pipe = encoder_pipeline(Path(source.model) if model is None else model, settings.device, VAE)
    out.mkdir(parents=True, exist_ok=True)
    lined: list[CacheEntry] = []
    for index, row in enumerate(rows):
        if index % settings.shards != settings.shard:
            continue
        if (out / row.entry).exists():
            with safe_open(str(out / row.entry), framework="pt") as done:
                lined.append(CacheEntry.model_validate_json(done.metadata()["entry"]))
            continue
        entry, tensors = region_crop_entry(pipe, cache, entries[row.entry], row, crops.parent)
        partial = out / f"{row.entry}.partial"
        save_file(tensors, str(partial), metadata={"entry": entry.model_dump_json()})
        partial.rename(out / row.entry)
        lined.append(entry)
        if (index + 1) % 100 == 0:
            console.print(f"{index + 1} of {len(rows)} entries given their region crops")
    if settings.shards > 1:
        return None
    (out / ENTRIES).write_text("".join(f"{entry.model_dump_json()}\n" for entry in lined), encoding="utf-8")
    forms = Counter(entry.variant.value for entry in lined)
    origin = RegionCropOrigin(
        cache=str(cache.resolve()),
        crops=str(crops.resolve()),
        place=build.place,
        region_crops=sum(len(row.crops) for row in rows),
        skipped=build.skipped,
    )
    record = source.model_copy(
        update={"entries": {form: forms[form] for form in source.entries}, "region_crops": origin}
    )
    record.write(out / CACHE_RECORD)
    return record


@app.command()
def main(
    cache: Annotated[Path, typer.Option(help="the finished diffu-moss-cache-pairs cache the crops were planned for")],
    crops: Annotated[
        Path, typer.Option(help="the manifest of region crops (diffu_page.data.region_crop_pairs' region_crops.jsonl)")
    ],
    out: Annotated[Path, typer.Option(help="the new cache's directory (a new one, or a stopped run's to resume)")],
    model: Annotated[
        Path | None,
        typer.Option(
            help="the MOSS model directory whose VAE encodes the crops (default: the one --cache was made with)"
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="where the VAE runs")] = "cuda",
    shard: Annotated[
        int, typer.Option(min=0, help="encode only this shard of --shards: the rows at shard, shard + shards, ...")
    ] = 0,
    shards: Annotated[
        int,
        typer.Option(
            min=1,
            help="split the rows into this many shards, each encoded by its own run into --out (one a GPU); a run "
            "without --shards over --out then finishes the cache",
        ),
    ] = 1,
) -> None:
    """Cache region crops in place of whole samples: each entry's crops encoded, its latents and prompt as cached."""
    try:
        settings = RegionCropCacheSettings(device=device, shard=shard, shards=shards)
        record = write_region_crop_cache(cache, crops, out, settings, model=model)
    except ValueError as error:  # the arguments' fault: a finished cache, a manifest of another cache, ...
        raise typer.BadParameter(str(error)) from error
    if record is None:
        console.print(f"shard {shard} of {shards} cached into {out}; run again without --shards to finish the cache")
        return
    origin = record.region_crops
    crops_held = 0 if origin is None else origin.region_crops
    console.print(f"{json.dumps(record.entries)} entries, {crops_held} region crops -> {out}")


if __name__ == "__main__":
    app()
