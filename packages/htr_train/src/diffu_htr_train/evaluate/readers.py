"""The readers scored: MiniCPM-OCR (a base or fine-tuned checkpoint) asked through a vLLM server as OpenBMB evaluates
it (``docs/benchmark_reproduction_en.md``: ``Text Recognition:`` after the image, :data:`MINICPM_SAMPLING`, at most 512
new tokens for a line, as on RIMES; the server sets ``downsample_mode`` 4x, ``max_slice_nums`` 9 and thinking off), and a TrOCR model (the Swedish Lion) read
with transformers as its model card reads it (``interpolate_pos_encoding``), greedy or with beams. Each reads the
lines in their order and times it (images loaded included, the model's loading not)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from diffu_htr_train.evaluate.lines import EvalLine

MINICPM_PROMPT = "Text Recognition:"
MINICPM_SAMPLING = {"temperature": 0.01, "top_p": 0.001, "top_k": 1, "repetition_penalty": 1.0}  # OpenBMB's, ~greedy
# sent with every request too, so a server started without them as defaults reads the same (OpenBMB's FAQ: thinking
# left on gives blank or garbled answers)
MINICPM_TEMPLATE = {"enable_thinking": False}
MINICPM_IMAGES = {"downsample_mode": "4x", "max_slice_nums": 9}


class MiniCPMSettings(BaseModel):
    """How MiniCPM-OCR is asked: the vLLM server, the model name it serves, requests at a time, new tokens at most."""

    model_config = {"frozen": True}

    endpoint: str = "http://localhost:8000"
    model: str
    concurrency: int = Field(default=32, ge=1)
    max_tokens: int = Field(default=512, ge=1)


class TrOCRSettings(BaseModel):
    """How a TrOCR model reads: its repo or folder, lines a batch, beams (1: greedy), new tokens at most (its training's
    128) and the weights' dtype (float32, as its own evaluation runs it, or bfloat16)."""

    model_config = {"frozen": True}

    model: str
    batch: int = Field(default=32, ge=1)
    beams: int = Field(default=1, ge=1)
    max_new_tokens: int = Field(default=128, ge=1)
    dtype: Literal["float32", "bfloat16"] = "float32"


class Reading(BaseModel):
    """A reader's prediction for each line, in the lines' order, why each ended (``stop``; ``length``: cut off at the
    token limit, a failure; None where the reader does not say) and the seconds reading them took."""

    predictions: list[str]
    finished: list[str | None]
    seconds: float


def read_minicpm(lines: list[EvalLine], settings: MiniCPMSettings) -> Reading:
    """Each line read by the vLLM server ``settings`` name, ``settings.concurrency`` requests at a time."""

    def body(line: EvalLine) -> dict[str, object]:
        image = {"type": "image_url", "image_url": {"url": Path(line.image).as_uri()}}
        return {
            "model": settings.model,
            **MINICPM_SAMPLING,
            "chat_template_kwargs": MINICPM_TEMPLATE,
            "mm_processor_kwargs": MINICPM_IMAGES,
            "max_tokens": settings.max_tokens,
            "messages": [{"role": "user", "content": [image, {"type": "text", "text": MINICPM_PROMPT}]}],
        }

    # httpx pools 100 connections by default: more requests at a time would wait on the pool, not the server
    limits = httpx.Limits(max_connections=settings.concurrency, max_keepalive_connections=settings.concurrency)
    with httpx.Client(base_url=settings.endpoint, timeout=600, limits=limits) as client:

        def read(line: EvalLine) -> tuple[str, str | None]:
            response = client.post("/v1/chat/completions", json=body(line))
            response.raise_for_status()
            choice = response.json()["choices"][0]
            return choice["message"]["content"] or "", choice.get("finish_reason")

        start = time.perf_counter()
        with ThreadPoolExecutor(settings.concurrency) as pool:
            answers = list(pool.map(read, lines))
        seconds = time.perf_counter() - start
        return Reading(
            predictions=[text for text, _ in answers], finished=[reason for _, reason in answers], seconds=seconds
        )


def read_trocr(lines: list[EvalLine], settings: TrOCRSettings) -> Reading:
    """Each line read by the TrOCR model ``settings`` name on the first GPU, a batch at a time."""
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    dtype = getattr(torch, settings.dtype)
    processor = TrOCRProcessor.from_pretrained(settings.model)
    reader = VisionEncoderDecoderModel.from_pretrained(settings.model, dtype=dtype).cuda().eval()
    # the Swedish Lion was trained to start decoding from its config's token (<bos>, 259); its generation_config.json
    # says 2 (<unk>), which transformers 5's generate takes, costing ~2 points of CER
    start_token = reader.config.decoder_start_token_id
    predictions: list[str] = []
    start = time.perf_counter()
    for first in range(0, len(lines), settings.batch):
        images = [Image.open(line.image).convert("RGB") for line in lines[first : first + settings.batch]]
        pixels = processor(images=images, return_tensors="pt").pixel_values.to("cuda", dtype)
        with torch.inference_mode():
            out = reader.generate(
                pixel_values=pixels,
                decoder_start_token_id=start_token,
                interpolate_pos_encoding=True,
                num_beams=settings.beams,
                max_new_tokens=settings.max_new_tokens,
            )
        predictions += processor.batch_decode(out, skip_special_tokens=True)
    return Reading(predictions=predictions, finished=[None] * len(predictions), seconds=time.perf_counter() - start)
