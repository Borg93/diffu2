"""diffu-moss-cache-pairs: style pairs or page pairs as a reference training reads them, encoded once on the GPU.

    diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/style_pairs_v3d/train.jsonl \\
        --out data_flux3/cache_moss_ref_v3d --limit 4000
    diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/style_pairs_v3d/val/val.jsonl \\
        --out data_flux3/cache_moss_ref_v3d_val --every-form
    diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/page_pairs/train.jsonl \\
        --out data_flux3/cache_moss_pages

The manifest is a style-pair set's or a page-pair set's (:func:`diffu_moss.train.pairs.read_manifest`): a page pair's
row carries its Layout prompt in every form, a style pair's caption is compiled into it here; both are cached alike.
The pairs whose reference repeats their target's lines are left out (:func:`diffu_moss.train.pairs.apart`); the rest
are shuffled by ``--seed`` and the first ``--limit`` kept, each in the prompt form the mix gives it
(:func:`diffu_moss.train.pairs.assign_variants`), or in all three (``--every-form``, a validation set). Each entry is
encoded exactly as drawing encodes it, by the MOSS model directory's own components loaded as the page pipeline loads
them (the text encoder in bfloat16, the VAE in float32): the target's Layout prompt by ``MossTextEncoderStep``, the
target and the reference by :func:`diffu_moss.latents.encode_latents` from :func:`diffu_moss.latents.image_pixels` (a
style pair's reference is already on the reference grid, as ``ReferenceGridStep`` puts one). Written under ``--out``:
one safetensors file per entry (``prompt_embeds``, ``target`` and, drawn with the sample, ``reference``, in bfloat16;
the entry's facts in its metadata), each written whole before it is named, so a stopped run resumes where it stopped;
then ``entries.jsonl`` and ``cache.json``, last.

With ``--latents-only`` the text encoder is not loaded and no entry holds ``prompt_embeds``: a training that encodes
each prompt in its step (``diffu-moss-train-ref --encode-text``, :class:`PromptEncoder`) reads such a cache; the
entries still give their prompts and token counts (by the tokenizer, as the encoder reads them).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated

import torch
import typer
from diffusers.modular_pipelines import ModularPipeline, SequentialPipelineBlocks
from PIL import Image
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from diffu.modular_pipelines.loading import COMPONENT_DTYPES, require_loaded
from diffu_moss.latents import encode_latents, image_pixels
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.modular_pipelines.moss.encoders import MossReferenceEncodeStep, MossTextEncoderStep, prompt_tokens
from diffu_moss.train.pairs import TrainingPair, Variant, apart, assign_variants, read_manifest
from diffu_moss.train.records import CACHE_RECORD, CacheRecord

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

ENTRIES = "entries.jsonl"
CACHE_DTYPE = torch.bfloat16
ENCODERS = ("text_encoder", "tokenizer", "vae")
LATENT_ENCODERS = ("tokenizer", "vae")  # a latents-only cache's: the tokenizer counts each prompt's tokens
TEXT_ENCODERS = ("text_encoder", "tokenizer")


class CacheEntry(BaseModel):
    """One cached training sample: a pair in one prompt form, its tensors in ``file``."""

    model_config = ConfigDict(frozen=True)

    file: str
    pair: str
    variant: Variant
    prompt: str
    tokens: int
    target_size: tuple[int, int]  # (width, height) px
    reference_size: tuple[int, int] | None


class CachedSample(BaseModel):
    """A cache entry's tensors: the prompt's embeddings ``[1, tokens, text_dim]`` (cached, or encoded as it is
    loaded), the target's latents ``[1, latent_channels, H / 16, W / 16]`` and the reference's (None for a sample drawn
    without it)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    entry: CacheEntry
    prompt_embeds: Tensor
    target: Tensor
    reference: Tensor | None


def encoder_pipeline(model: Path, device: str, names: tuple[str, ...] = ENCODERS) -> ModularPipeline:
    """The encoder steps of the MOSS model directory ``model`` (its text encoder, and its reference encoder for the
    VAE it names; called without an image, it encodes none), with its components ``names`` (by default its text
    encoder, tokenizer and VAE) loaded as the page pipeline loads them (``COMPONENT_DTYPES``), on ``device``.

    Raises:
        RuntimeError: If a component failed to load.
    """
    blocks = SequentialPipelineBlocks.from_blocks_dict(
        {"text_encoder": MossTextEncoderStep(), "reference_encoder": MossReferenceEncodeStep()}
    )
    pipe = MossModularPipeline(blocks=blocks, pretrained_model_name_or_path=str(model))
    pipe.load_components(names=list(names), dtype=COMPONENT_DTYPES)
    require_loaded(pipe, names)
    pipe.to(device)
    return pipe


class PromptEncoder:
    """The frozen text encoder of the MOSS model directory ``model`` on ``device``, loaded as the cache loads it
    (:func:`encoder_pipeline`), encoding a prompt in inference mode exactly as the cache encodes it: a training's step
    reads its prompts from it (``--encode-text``) instead of from cached embeddings.

    Attributes:
        pipe: The encoder pipeline, its text encoder frozen and in eval mode.
    """

    def __init__(self, model: Path, device: str) -> None:
        self.pipe = encoder_pipeline(model, device, TEXT_ENCODERS)
        self.pipe.text_encoder.requires_grad_(False).eval()

    def encode(self, prompt: str) -> Tensor:
        """The prompt's embeddings ``[1, tokens, text_dim]`` as a cache entry holds them (in the encoder's bfloat16), on
        the encoder's device: a plain tensor, which the DiT's backward may save."""
        with torch.inference_mode():
            embeds = self.pipe(prompt=prompt, output="prompt_embeds")
        return embeds.clone()


def image_latents(pipe: ModularPipeline, path: Path) -> tuple[Tensor, tuple[int, int]]:
    """The image at ``path`` as the DiT's latents, as the reference encode step encodes a reference, and its size."""
    with Image.open(path) as image:
        return encode_latents(pipe.vae, image_pixels(image)), image.size


def cache_entry(
    pipe: ModularPipeline, pair: TrainingPair, variant: Variant, file: str, *, latents_only: bool = False
) -> tuple[CacheEntry, dict[str, Tensor]]:
    """``pair`` in ``variant``'s prompt form, encoded: its entry and its tensors (in :data:`CACHE_DTYPE`, on the CPU;
    with ``latents_only`` no ``prompt_embeds``).

    Raises:
        ValueError: If the pair's caption cannot be compiled in that form, or an image is off the latent grid.
    """
    target, target_size = image_latents(pipe, pair.target)
    prompt = pair.prompt(variant, target_size)
    tensors = {"target": target}
    if latents_only:
        tokens = prompt_tokens(pipe.tokenizer, prompt)
    else:
        tensors["prompt_embeds"] = pipe(prompt=prompt, output="prompt_embeds")
        tokens = tensors["prompt_embeds"].shape[1]
    reference_size = None
    if variant.takes_reference:
        tensors["reference"], reference_size = image_latents(pipe, pair.reference)
    entry = CacheEntry(
        file=file,
        pair=pair.id,
        variant=variant,
        prompt=prompt,
        tokens=tokens,
        target_size=target_size,
        reference_size=reference_size,
    )
    return entry, {name: tensor.to("cpu", CACHE_DTYPE).contiguous() for name, tensor in tensors.items()}


def planned_entries(
    pairs: list[TrainingPair], *, limit: int | None, seed: int, every_form: bool
) -> list[tuple[TrainingPair, Variant]]:
    """The pairs to cache, each with the prompt form it is cached in: the pairs shuffled by ``seed``, the first
    ``limit`` of them (all when None), each in the form the mix gives it, or in every form."""
    chosen = list(pairs)
    random.Random(seed).shuffle(chosen)
    chosen = chosen[:limit]
    if every_form:
        return [(pair, variant) for pair in chosen for variant in Variant]
    return list(zip(chosen, assign_variants(len(chosen), seed), strict=True))


class CacheSettings(BaseModel):
    """Which pairs a cache holds, in which prompt forms, and where it is encoded."""

    model_config = ConfigDict(frozen=True)

    limit: int | None = None
    seed: int = 0
    every_form: bool = False
    latents_only: bool = False
    device: str = "cuda"


def write_cache(model: Path, manifest: Path, out: Path, settings: CacheSettings) -> CacheRecord:
    """The pairs of ``manifest`` cached into ``out`` by the MOSS model directory ``model``'s encoders (the module's
    docstring), resuming the entries a stopped run wrote.

    Returns:
        The cache's record.

    Raises:
        ValueError: If ``out`` holds a finished cache already, or a pair's caption cannot be compiled.
        RuntimeError: If an encoder failed to load.
    """
    if (out / CACHE_RECORD).exists():
        raise ValueError(f"{out} holds a finished cache already; give a new directory")
    pairs = read_manifest(manifest)
    kept = apart(pairs)
    planned = planned_entries(kept, limit=settings.limit, seed=settings.seed, every_form=settings.every_form)
    pipe = encoder_pipeline(model, settings.device, LATENT_ENCODERS if settings.latents_only else ENCODERS)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (pair, variant) in enumerate(planned):
        file = f"{index:06d}_{variant}.safetensors"
        if (out / file).exists():
            with safe_open(str(out / file), framework="pt") as written:
                entries.append(CacheEntry.model_validate_json(written.metadata()["entry"]))
            continue
        entry, tensors = cache_entry(pipe, pair, variant, file, latents_only=settings.latents_only)
        partial = out / f"{file}.partial"
        save_file(tensors, str(partial), metadata={"entry": entry.model_dump_json()})
        partial.rename(out / file)
        entries.append(entry)
        if (index + 1) % 100 == 0:
            console.print(f"{index + 1} of {len(planned)} entries cached")
    (out / ENTRIES).write_text("".join(f"{entry.model_dump_json()}\n" for entry in entries), encoding="utf-8")
    counts = {variant.value: sum(entry.variant is variant for entry in entries) for variant in Variant}
    record = CacheRecord(
        pairs=str(manifest.resolve()),
        model=str(model.resolve()),
        seed=settings.seed,
        pairs_read=len(pairs),
        pairs_sharing_lines=len(pairs) - len(kept),
        entries=counts,
        latents_only=settings.latents_only,
    )
    record.write(out / CACHE_RECORD)
    return record


def read_entries(cache: Path) -> list[CacheEntry]:
    """The entries of the finished cache at ``cache``.

    Raises:
        ValueError: If ``cache`` holds no finished cache.
    """
    if not (cache / CACHE_RECORD).is_file():
        raise ValueError(f"{cache} holds no finished cache (no {CACHE_RECORD}): run diffu-moss-cache-pairs")
    with (cache / ENTRIES).open(encoding="utf-8") as rows:
        return [CacheEntry.model_validate_json(row) for row in rows if row.strip()]


def with_reference(entries: list[CacheEntry], count: int) -> list[CacheEntry]:
    """The first ``count`` of ``entries`` drawn with their reference, in the cache's order: the fixed handful of pairs
    an overfit test trains on (``diffu-moss-train-ref --limit-pairs``) and draws back."""
    return [entry for entry in entries if entry.variant.takes_reference][:count]


def read_record(cache: Path) -> CacheRecord:
    """The record of the finished cache at ``cache``.

    Raises:
        ValueError: If ``cache`` holds no finished cache.
    """
    if not (cache / CACHE_RECORD).is_file():
        raise ValueError(f"{cache} holds no finished cache (no {CACHE_RECORD}): run diffu-moss-cache-pairs")
    return CacheRecord.read(cache / CACHE_RECORD)


def load_sample(
    cache: Path, entry: CacheEntry, device: torch.device | str, encoder: PromptEncoder | None = None
) -> CachedSample:
    """An entry's tensors, on ``device``: its prompt's embeddings encoded by ``encoder`` when given, else cached.

    Raises:
        ValueError: If no encoder is given and the entry holds no prompt embeddings (a latents-only cache).
    """
    tensors = load_file(str(cache / entry.file), device=str(device))
    if encoder is not None:
        tensors["prompt_embeds"] = encoder.encode(entry.prompt).to(device)
    elif "prompt_embeds" not in tensors:
        raise ValueError(f"{cache} holds no prompt embeddings (a latents-only cache): encode them (--encode-text)")
    return CachedSample(
        entry=entry,
        prompt_embeds=tensors["prompt_embeds"],
        target=tensors["target"],
        reference=tensors.get("reference"),
    )


@app.command()
def main(
    model: Annotated[
        Path, typer.Option(help="a MOSS model directory (diffu-moss-init): its text encoder, tokenizer, VAE")
    ],
    pairs: Annotated[
        Path,
        typer.Option(help="a style-pair manifest (style_pairs' train.jsonl or val/val.jsonl) or a page_pairs one"),
    ],
    out: Annotated[Path, typer.Option(help="the cache's directory (a new one, or a stopped run's to resume)")],
    limit: Annotated[
        int | None, typer.Option(min=1, help="cache this many pairs, picked by --seed (default: all)")
    ] = None,
    seed: Annotated[int, typer.Option(help="picks the pairs and their prompt forms")] = 0,
    every_form: Annotated[
        bool, typer.Option("--every-form", help="cache every pair in all three prompt forms (a validation set)")
    ] = False,
    latents_only: Annotated[
        bool,
        typer.Option(
            "--latents-only",
            help="cache the latents alone, no prompt embeddings (a training then encodes them: --encode-text)",
        ),
    ] = False,
    device: Annotated[str, typer.Option(help="where the encoders run")] = "cuda",
) -> None:
    """Cache style pairs for a reference training: each pair's prompt embeddings and latents."""
    settings = CacheSettings(limit=limit, seed=seed, every_form=every_form, latents_only=latents_only, device=device)
    try:
        record = write_cache(model, pairs, out, settings)
    except ValueError as error:  # the arguments' fault: a finished cache, or a manifest that is not a described set
        raise typer.BadParameter(str(error)) from error
    left_out = f"{record.pairs_sharing_lines} of {record.pairs_read} pairs left out for sharing a line"
    console.print(f"{json.dumps(record.entries)} entries ({left_out}) -> {out}")


if __name__ == "__main__":
    app()
