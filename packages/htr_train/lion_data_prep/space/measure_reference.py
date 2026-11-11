"""The Space's reference timings: each example region read on this machine by the Swedish Lion TrOCR (transformers,
its lines one after the other and as one batch, as the app runs it) and by the reader in vLLM, alone and with its
DSpark draft, one request at a time. The app shows them beside its live race, which runs in PyTorch because ZeroGPU
cannot host vLLM.

Serve the reader twice first, as the draft's card says (prefix caching off), the second server once the first
answers, on a GPU nothing else is using:

    vllm serve READER --served-model-name htr ... --port 8057
    vllm serve READER --served-model-name htr ... --port 8056 --speculative-config '{"method": "dspark", ...}'
    python measure_reference.py --examples examples --gpu "RTX PRO 6000 Blackwell" --out reference.json

Each reading is timed from the encoded images to the reading: the median of RUNS runs after one that is not timed.
vLLM keeps the processed form of an image it has seen, so every request sends the region with one pixel, drawn at
random, changed by one level: the region is new to the server each time, as a region is in production.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import secrets
import statistics
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from reference import Reference, RegionTimes

SAMPLING = {"temperature": 0.01, "top_p": 0.001, "top_k": 1, "repetition_penalty": 1.0}  # OpenBMB's, near greedy
IMAGES = {"downsample_mode": "4x", "max_slice_nums": 9}


def median_seconds(read: Callable[[int], None], runs: int) -> float:
    """The median time of `runs` calls of `read`, after one that is not timed; each call gets its own number."""
    read(0)
    times = []
    for run in range(1, runs + 1):
        started = time.perf_counter()
        read(run)
        times.append(time.perf_counter() - started)
    return round(statistics.median(times), 3)


def requests_for(region: Path, count: int) -> list[dict[str, Any]]:
    """`count` requests for one region, each with one pixel, drawn at random, changed by one level."""
    image = Image.open(region).convert("RGB")
    pixels = image.tobytes()
    bodies = []
    for _ in range(count):
        changed = bytearray(pixels)
        changed[3 * secrets.randbelow(image.width * image.height)] ^= 1  # a pixel's red
        buffer = io.BytesIO()
        Image.frombytes("RGB", image.size, bytes(changed)).save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Text Recognition:"}]
        bodies.append(
            {
                "model": "htr",
                **SAMPLING,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
                "mm_processor_kwargs": IMAGES,
                "messages": [{"role": "user", "content": content}],
            }
        )
    return bodies


class Vllm:
    """One vLLM server, asked for one region at a time; it keeps what the region's requests answered."""

    def __init__(self, endpoint: str, client: httpx.Client) -> None:
        self.url = f"{endpoint}/v1/chat/completions"
        self.client = client
        self.texts: set[str] = set()
        self.tokens = 0

    def read(self, body: dict[str, Any]) -> None:
        reply = self.client.post(self.url, json=body)
        reply.raise_for_status()
        answer = reply.json()
        self.texts.add(answer["choices"][0]["message"]["content"])
        self.tokens = answer["usage"]["completion_tokens"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", type=Path, required=True, help="the folder build_examples.py wrote")
    parser.add_argument("--trocr", default="Gabriel/swedish-lion-trocr-large-handwritten-char")
    parser.add_argument("--alone", default="http://localhost:8057", help="vLLM serving the reader alone")
    parser.add_argument("--draft", default="http://localhost:8056", help="vLLM serving the reader with its draft")
    parser.add_argument("--gpu", required=True, help="the GPU's name, as the app will show it")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained(args.trocr)
    trocr = VisionEncoderDecoderModel.from_pretrained(args.trocr).to("cuda").eval()

    def read_lines(files: list[bytes], batch: int) -> None:
        lines = [Image.open(io.BytesIO(data)).convert("RGB") for data in files]
        for first in range(0, len(lines), batch):
            pixels = processor(images=lines[first : first + batch], return_tensors="pt").pixel_values.to("cuda")
            with torch.inference_mode():
                trocr.generate(
                    pixels,
                    decoder_start_token_id=trocr.config.decoder_start_token_id,
                    interpolate_pos_encoding=True,
                    num_beams=1,
                    max_new_tokens=128,
                )
        torch.cuda.synchronize()

    regions = {}
    with httpx.Client(timeout=600) as client:
        for example in json.loads((args.examples / "examples.json").read_text(encoding="utf-8")):
            folder = args.examples / example["id"]
            files = [(folder / f"line_{i:02d}.jpg").read_bytes() for i in range(len(example["lines"]))]
            bodies = requests_for(folder / "region.png", args.runs + 1)
            alone, draft = Vllm(args.alone, client), Vllm(args.draft, client)
            times = RegionTimes(
                lines=len(files),
                trocr_single=median_seconds(lambda _, files=files: read_lines(files, 1), args.runs),
                trocr_batch=median_seconds(lambda _, files=files: read_lines(files, len(files)), args.runs),
                vllm=median_seconds(lambda run, alone=alone, bodies=bodies: alone.read(bodies[run]), args.runs),
                vllm_dspark=median_seconds(lambda run, draft=draft, bodies=bodies: draft.read(bodies[run]), args.runs),
                tokens=alone.tokens,
            )
            regions[example["id"]] = times
            same = "the same text" if alone.texts == draft.texts and len(alone.texts) == 1 else "texts differ"
            print(example["id"], times.model_dump(), f"{len(alone.texts)} and {len(draft.texts)} readings, {same}")
    reference = Reference(gpu=args.gpu, measured=date.today().isoformat(), runs=args.runs, regions=regions)
    args.out.write_text(reference.model_dump_json(indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
