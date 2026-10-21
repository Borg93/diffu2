"""Reading generated lines back with a Swedish HTR model, and the error rates that score them."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from page_synth.captions.prompt import LineRecord

HTR = "Riksarkivet/trocr-base-handwritten-hist-swe-2"


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    return levenshtein(pred, ref) / max(len(ref), 1)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def pixel_box(bbox: list[int], size: tuple[int, int]) -> tuple[float, float, float, float]:
    """A layout box (``[y_min, x_min, y_max, x_max]`` on 0-1000) as ``(x0, y0, x1, y1)`` pixels of ``size``."""
    width, height = size
    y0, x0, y1, x1 = bbox
    return x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000


def materialize_position_tables(model: VisionEncoderDecoderModel) -> None:
    """Rebuild TrOCR's sinusoidal position tables that a meta-device load left on meta, on the model's device.

    TrOCR keeps the table as a plain attribute built in ``__init__``, not a buffer, so ``from_pretrained`` (which
    initializes on the meta device) never fills it and the first forward fails with "Cannot copy out of meta tensor".
    """
    for module in model.modules():
        if type(module).__name__ == "TrOCRSinusoidalPositionalEmbedding" and module.weights.is_meta:
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
