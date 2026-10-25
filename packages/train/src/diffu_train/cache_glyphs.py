"""Crops cached with their glyph maps for diffu's E2, in NeMo's image-edit cache layout: the crop is the target, its
glyph map (``diffu_page.glyphs``) the one reference image, its caption the prompt.

Every tensor is made by diffu's own encoders (``diffu_train.cache``), so training sees what generation will when the
pipeline is given the glyph map as its reference image. The files are laid out as NeMo's ``ImageEditDataset`` reads
them, field for field as its own FLUX.3 cache writer lays them out: ``samples/sample_NNNNNNNN.pt``,
``metadata_shard_NNNN.json`` listing each with its compound bucket signature and token lengths, ``metadata.json``
naming the shards, written last so it marks a finished cache; beside them ``empty_prompt.pt`` for caption dropout.

    python -m diffu_train.cache_glyphs --crops <split dir> --model <diffu repo> --out <cache dir> [--scale 0.5]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import torch
import typer
from pydantic import BaseModel

from diffu_page.captions.records import CropRecord, read_records
from diffu_page.glyphs import glyph_map
from diffu_page.output import console, err_console
from diffu_train.cache import SHARD_SIZE, DiffuEncoders, encode_image, encode_prompt, window, write_empty_prompt

SAMPLES = "samples"

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class Signature(BaseModel):
    """A pair's latent shapes, ``(C, H, W)``: NeMo buckets pairs by them."""

    target: tuple[int, int, int]
    contexts: list[tuple[int, int, int]]


class PairEntry(BaseModel):
    """A pair's row in a metadata shard, as NeMo's ``ImageEditDataset`` reads it."""

    cache_file: str  # relative to the cache dir
    compound_bucket_signature: Signature
    context_token_lengths: list[int]
    original_id: str
    row_index: int
    target_token_length: int
    text_token_length: int


class PairRun(BaseModel):
    """A cached split: every pair that was cached, and every crop that failed with its error."""

    entries: list[PairEntry]
    failed: dict[str, str]


def _tokens(latent: torch.Tensor) -> int:
    return latent.shape[-2] * latent.shape[-1]


def _cache_pair(
    index: int, record: CropRecord, crops_dir: Path, *, encoders: DiffuEncoders, out: Path, scale: float
) -> PairEntry:
    image = window(crops_dir, record, scale=scale)
    target = encode_image(encoders, image)
    context = encode_image(encoders, glyph_map(record.lines, image.size))
    embeddings = encode_prompt(encoders, record.caption)
    signature = Signature(target=tuple(target.shape), contexts=[tuple(context.shape)])
    entry = PairEntry(
        cache_file=f"{SAMPLES}/sample_{index:08d}.pt",
        compound_bucket_signature=signature,
        context_token_lengths=[_tokens(context)],
        original_id=Path(record.file_name).stem,
        row_index=index,
        target_token_length=_tokens(target),
        text_token_length=embeddings.shape[0],
    )
    metadata = {
        "original_ids": {"id": entry.original_id, "row_index": index},
        "target_spatial_shape": list(target.shape[-2:]),
        "context_spatial_shapes": [list(context.shape[-2:])],
        "target_token_length": entry.target_token_length,
        "context_token_lengths": entry.context_token_lengths,
        "text_token_length": entry.text_token_length,
        "compound_bucket_signature": signature.model_dump(mode="json"),
        "conditioning_shapes": {},
        "manifest_metadata": {"row_index": index, "file_name": record.file_name, "scale": scale},
    }
    sample = {
        "target_latent": target,
        "context_latents": [context],
        "prompt_embeddings": embeddings,
        "prompt_attention_mask": torch.ones(embeddings.shape[0], dtype=torch.long),  # the transformer sees every token
        "conditioning_tensors": {},
        "metadata": metadata,
    }
    torch.save(sample, out / entry.cache_file)
    return entry


def cache_glyph_pairs(
    records: list[CropRecord], crops_dir: Path, *, encoders: DiffuEncoders, out: Path, scale: float = 1.0
) -> PairRun:
    """Every crop of a split with its glyph map, scaled by ``scale``, cached into ``out``; a crop that fails is
    recorded with its error and the rest are still cached."""
    (out / SAMPLES).mkdir(parents=True, exist_ok=True)
    entries: list[PairEntry] = []
    failed: dict[str, str] = {}
    for record in records:
        try:
            entries.append(_cache_pair(len(entries), record, crops_dir, encoders=encoders, out=out, scale=scale))
        except Exception as error:  # one unreadable crop must not end a cache of thousands
            failed[record.file_name] = f"{type(error).__name__}: {error}"
    return PairRun(entries=entries, failed=failed)


def write_pair_metadata(entries: list[PairEntry], out: Path, *, model: Path, scale: float) -> None:
    """``out``'s metadata shards and ``metadata.json``, as NeMo's ``ImageEditDataset`` loads them. ``metadata.json`` is
    written last, so it marks a finished cache.

    Raises:
        ValueError: If ``entries`` is empty: a cache that holds nothing is not finished.
    """
    if not entries:
        raise ValueError(f"nothing was cached into {out}: no metadata.json marks it finished")
    shards: list[str] = []
    for start in range(0, len(entries), SHARD_SIZE):
        name = f"metadata_shard_{start // SHARD_SIZE:04d}.json"
        rows = [entry.model_dump(mode="json") for entry in entries[start : start + SHARD_SIZE]]
        (out / name).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        shards.append(name)
    metadata = {
        "preprocessing_config": {"processor": "diffu_train.cache_glyphs", "model_name": str(model), "scale": scale},
        "row_limit": len(entries),
        "shards": shards,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@app.command()
def main(
    crops: Annotated[Path, typer.Option(help="a crop split: metadata.jsonl and its images")],
    model: Annotated[Path, typer.Option(help="the diffu repo whose encoders make the cache")],
    out: Annotated[Path, typer.Option(help="the cache directory")],
    scale: Annotated[float, typer.Option(min=0.1, max=1.0, help="scale the crops (0.5: the 512-px stage)")] = 1.0,
    limit: Annotated[int | None, typer.Option(min=1, help="cache only the first N crops (a smoke test)")] = None,
    device: Annotated[str, typer.Option(help="where the encoders run")] = "cuda",
) -> None:
    """Cache a split's crops with their glyph maps for diffu: target, glyph-map and prompt encodings."""
    records = read_records(crops / "metadata.jsonl")[:limit]
    out.mkdir(parents=True, exist_ok=True)
    encoders = DiffuEncoders.load(model, device)
    write_empty_prompt(encoders, out)  # first: seconds, where the crops take hours
    run = cache_glyph_pairs(records, crops, encoders=encoders, out=out, scale=scale)
    (out / "failed.json").write_text(json.dumps(run.failed, indent=1), encoding="utf-8")
    if run.failed:
        err_console.print(f"{len(run.failed)} of {len(records)} crops failed: see {out / 'failed.json'}")
    if not run.entries:  # no metadata.json: it marks a finished cache
        raise typer.Exit(code=1)
    write_pair_metadata(run.entries, out, model=model, scale=scale)
    console.print(f"{len(run.entries)} pairs -> {out}")


if __name__ == "__main__":
    app()
