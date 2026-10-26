"""Reading drawn lines back with a Swedish HTR model, and the error rates that score them."""

from __future__ import annotations

import re

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding

from diffu_page.captions.prompt import LineRecord
from diffu_page.crops.geometry import pixel_box


def levenshtein(a: str, b: str) -> int:
    """The edit distance between two strings: insertions, deletions and substitutions, one each."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """The character error rate of ``pred`` against ``ref``: their edit distance per reference character."""
    return levenshtein(pred, ref) / max(len(ref), 1)


def numbers_differ(pred: str, ref: str) -> bool:
    """Whether ``pred`` reads other numbers than ``ref`` holds (each run of digits, in order): one wrong digit in a
    date, an amount or a page number makes a label wrong however small the line's CER."""
    return re.findall(r"\d+", pred) != re.findall(r"\d+", ref)


def materialize_position_tables(model: VisionEncoderDecoderModel) -> None:
    """Rebuild TrOCR's sinusoidal position tables that a meta-device load left on meta, on the model's device.

    TrOCR keeps the table as a plain attribute built in ``__init__``, not a buffer, so ``from_pretrained`` (which
    initializes on the meta device) never fills it and the first forward fails with "Cannot copy out of meta tensor".
    """
    for module in model.modules():
        if isinstance(module, TrOCRSinusoidalPositionalEmbedding) and module.weights.is_meta:
            table = module.get_embedding(module.weights.shape[0], module.embedding_dim, module.padding_idx)
            module.weights = table.to(model.device, model.dtype)


@torch.no_grad()
def read_lines(
    image: Image.Image, lines: list[LineRecord], *, model: VisionEncoderDecoderModel, processor: TrOCRProcessor
) -> tuple[list[str], list[float]]:
    """Read each line's box of ``image``: the reads and their CERs against the lines' texts."""
    crops = [image.crop(pixel_box(line.bbox, image.size)).convert("RGB") for line in lines]
    pixels = processor(images=crops, return_tensors="pt").pixel_values
    ids = model.generate(pixels.to(model.device, model.dtype), max_new_tokens=128)
    reads = [read.strip() for read in processor.batch_decode(ids, skip_special_tokens=True)]
    return reads, [cer(read, line.text) for read, line in zip(reads, lines, strict=True)]
