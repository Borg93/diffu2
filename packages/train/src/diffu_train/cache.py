"""Crops cached for training diffu with NeMo AutoModel, in NeMo's text-to-image cache layout.

Each crop's latents (``diffu.latents.encode_latents``) and prompt embeddings
(``diffu.modular_pipelines.diffu.encoders.prompt_embeds``) are made by the functions diffu's own pipeline runs, so
training sees what generation will: the same Qwen3.5 layers and 80-token windows, the same Qwen-Image VAE latents and
normalization, the same image preprocessing (FLUX.3's image processor). The files are laid out as NeMo's
``TextToImageDataset`` reads them: ``<out>/<W>x<H>/<md5>.pt`` per crop, ``metadata_shard_NNNN.json`` listing them,
and ``metadata.json`` naming the shards. ``diffu_train.cache_diffu`` is the command that writes a split.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict

from diffu.latents import encode_latents
from diffu.modular_pipelines.diffu import DiffuModularPipeline
from diffu.modular_pipelines.diffu.encoders import MAX_SEQUENCE_LENGTH, prompt_embeds
from diffu.modular_pipelines.loading import COMPONENT_DTYPES, require_loaded
from diffu_page.captions.records import CropRecord

SHARD_SIZE = 1000  # crops per metadata shard, as NeMo's own cache writer shards them
ENCODERS = ["text_encoder", "tokenizer", "vae"]  # the components caching loads; the transformer is not one

type BucketId = Callable[[int, int], int]


class CacheEntry(BaseModel):
    """A crop's row in a metadata shard: what NeMo's ``TextToImageDataset`` reads to find and bucket it."""

    cache_file: str
    image_path: str
    bucket_resolution: tuple[int, int]
    original_resolution: tuple[int, int]
    prompt: str
    bucket_id: int
    aspect_ratio: float
    pixels: int
    model_type: str = "diffu"


class CacheRun(BaseModel):
    """A cached split: every crop that was cached, and every crop that failed with its error."""

    entries: list[CacheEntry]
    failed: dict[str, str]


class DiffuEncoders(BaseModel):
    """diffu's text encoder, tokenizer, VAE and image processor, and the text layers its repo names, loaded once."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text_encoder: Any
    tokenizer: Any
    vae: Any
    image_processor: Any
    layers: tuple[int, ...]
    device: str

    @classmethod
    def load(cls, model: Path, device: str) -> Self:
        """The encoders of the diffu repo at ``model`` (VAE in float32, the text encoder in bfloat16) on ``device``.

        Raises:
            RuntimeError: If one of them failed to load, before a single crop fails on it.
        """
        pipe = DiffuModularPipeline.from_pretrained(str(model))
        pipe.load_components(names=ENCODERS, dtype=COMPONENT_DTYPES)
        require_loaded(pipe, ENCODERS)
        pipe.to(device)
        return cls(
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            vae=pipe.vae,
            image_processor=pipe.image_processor,
            layers=tuple(pipe.config.text_layers),
            device=device,
        )


@torch.no_grad()
def encode_crop(encoders: DiffuEncoders, image: Image.Image, caption: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A crop's latents ``(C, H/16, W/16)`` in float32 and its prompt embeddings ``(window, width)``, on the CPU."""
    pixels, _ = encoders.image_processor.preprocess(image.convert("RGB"), resize_mode="fill")
    latents = encode_latents(encoders.vae, pixels.to(encoders.device, encoders.vae.dtype))
    embeds = prompt_embeds(
        encoders.text_encoder,
        encoders.tokenizer,
        caption,
        layers=encoders.layers,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        device=torch.device(encoders.device),
    )
    return latents[0].float().cpu(), embeds[0].cpu()


def _cache_crop(
    record: CropRecord, crops_dir: Path, *, encoders: DiffuEncoders, out: Path, bucket_id: BucketId
) -> CacheEntry:
    image_path = (crops_dir / record.file_name).absolute()
    with Image.open(image_path) as image:
        latent, embeds = encode_crop(encoders, image, record.caption)
        width, height = image.size
    resolution = f"{width}x{height}"
    cache_file = out / resolution / f"{hashlib.md5(f'{image_path}_{resolution}'.encode()).hexdigest()}.pt"
    entry = CacheEntry(
        cache_file=str(cache_file),
        image_path=str(image_path),
        bucket_resolution=(width, height),
        original_resolution=(width, height),
        prompt=record.caption,
        bucket_id=bucket_id(width, height),
        aspect_ratio=width / height,
        pixels=width * height,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "latent": latent,
        "prompt_embeds": embeds,
        "original_resolution": entry.original_resolution,
        "bucket_resolution": entry.bucket_resolution,
        "crop_offset": (0, 0),  # crops are saved at their bucket's exact size: nothing is cut
        "prompt": entry.prompt,
        "image_path": entry.image_path,
        "bucket_id": entry.bucket_id,
        "aspect_ratio": entry.aspect_ratio,
        "model_type": entry.model_type,
    }
    torch.save(sample, cache_file)
    return entry


def cache_crops(
    records: list[CropRecord], crops_dir: Path, *, encoders: DiffuEncoders, out: Path, bucket_id: BucketId
) -> CacheRun:
    """Every crop of a split (its records, its images in ``crops_dir``) cached into ``out``; a crop that fails is
    recorded with its error and the rest are still cached."""
    entries: list[CacheEntry] = []
    failed: dict[str, str] = {}
    for record in records:
        try:
            entries.append(_cache_crop(record, crops_dir, encoders=encoders, out=out, bucket_id=bucket_id))
        except Exception as error:  # one unreadable crop must not end a cache of thousands
            failed[record.file_name] = f"{type(error).__name__}: {error}"
    return CacheRun(entries=entries, failed=failed)


def write_metadata(entries: list[CacheEntry], out: Path, *, model: Path) -> None:
    """``out``'s metadata shards and ``metadata.json``, as NeMo's ``TextToImageDataset`` loads them. ``metadata.json``
    is written last, so it marks a finished cache.

    Raises:
        ValueError: If ``entries`` is empty: a cache that holds nothing is not finished.
    """
    if not entries:
        raise ValueError(f"nothing was cached into {out}: no metadata.json marks it finished")
    shards: list[str] = []
    for start in range(0, len(entries), SHARD_SIZE):
        name = f"metadata_shard_{start // SHARD_SIZE:04d}.json"
        rows = [entry.model_dump(mode="json") for entry in entries[start : start + SHARD_SIZE]]
        (out / name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        shards.append(name)
    metadata = {
        "processor": "diffu",
        "model_name": str(model),
        "model_type": "diffu",
        "total_items": len(entries),
        "num_shards": len(shards),
        "shard_size": SHARD_SIZE,
        "shards": shards,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
