"""Jev's judgement of a scan's labels: each labelled line cut as Jev's training lines were, its label's text drawn
beside it, scored for "the text matches the line", and the lines' scores gathered for the scan.

Run with ``envs/vllm``'s Python (it holds vLLM, not the project's other packages, so the ALTO file is read here)::

    CUDA_VISIBLE_DEVICES=2 packages/htr_train/envs/vllm/.venv/bin/python -m diffu_htr_train.jev.score \
        PAIRS.tsv OUT_DIR --model runs/htr/final/hub/diffu-minicpm-ocr-htr-jev [--limit N]

``PAIRS.tsv`` holds a scan's image and its ALTO file a line, tab-separated. ``OUT_DIR`` gets JSON-lines parts, one
row a scan (:class:`ScanScore`), and a scan a part holds is skipped, so a stopped run goes on where it was.

This Jev tells a gross mismatch (another line's text, a mangled one: AUROC 0.99) from a match, and hardly a small
misreading (0.62): a low score says the label is not this line's text, a high one does not say it is letter-perfect.
"""

from __future__ import annotations

import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from diffu_htr_train.jev.prompt import model_prompt
from diffu_htr_train.jev.render import render_text

ALTO = "{http://www.loc.gov/standards/alto/ns-v4#}"
HEIGHT = 48  # the height Jev's training crops were scaled to (jev_lines' options)
MIN_SIDE = 10  # a line's box must be this many pixels on its shorter side, as the training lines' rule
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
MATCH = 0.5  # the score from which Jev calls a pair a match
PART = 5_000  # scans a part
SCANS_A_BATCH = 64

Image.MAX_IMAGE_PIXELS = None

type Point = tuple[float, float]


class Line(BaseModel):
    """A labelled line: its text, and its outline in the scan's pixels (its box's corners where the file has none)."""

    model_config = ConfigDict(frozen=True)

    text: str
    outline: tuple[Point, ...]

    @property
    def box(self) -> tuple[int, int, int, int]:
        xs, ys = [x for x, _ in self.outline], [y for _, y in self.outline]
        return math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))


class ScanScore(BaseModel):
    """What Jev makes of a scan's labels.

    Attributes:
        image: The scan.
        lines: How many labelled lines were scored.
        mean: The mean of the lines' scores (None for a scan with no line to score).
        low: The share of the lines Jev calls no match (a score under :data:`MATCH`).
        worst: The lowest score.
    """

    model_config = ConfigDict(frozen=True)

    image: str
    lines: int
    mean: float | None
    low: float | None
    worst: float | None


def alto_lines(path: Path) -> list[Line]:
    """The lines of an ALTO file that hold text and are large enough to hold writing, in the file's order."""
    lines: list[Line] = []
    for element in ET.parse(path).getroot().iter(f"{ALTO}TextLine"):
        text = " ".join(word.get("CONTENT", "") for word in element.iter(f"{ALTO}String")).strip()
        polygon = element.find(f"{ALTO}Shape/{ALTO}Polygon")
        numbers = [
            float(n) for n in (polygon.get("POINTS", "") if polygon is not None else "").replace(",", " ").split()
        ]
        outline = tuple(zip(numbers[0::2], numbers[1::2], strict=False))
        if len(outline) < 3:
            x, y = float(element.get("HPOS", 0)), float(element.get("VPOS", 0))
            across, down = float(element.get("WIDTH", 0)), float(element.get("HEIGHT", 0))
            outline = ((x, y), (x + across, y), (x + across, y + down), (x, y + down))
        line = Line(text=text, outline=outline)
        x0, y0, x1, y1 = line.box
        if text and min(x1 - x0, y1 - y0) >= MIN_SIDE:
            lines.append(line)
    return lines


def line_crop(scan: Image.Image, line: Line) -> Image.Image:
    """The line cut from ``scan`` as Jev's training lines were: its box, white outside its outline, :data:`HEIGHT`
    pixels high."""
    x0, y0, x1, y1 = line.box
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(scan.width, x1), min(scan.height, y1)
    crop = scan.crop((x0, y0, x1, y1)).convert("RGB")
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in line.outline], fill=255)
    crop = Image.composite(crop, Image.new("RGB", crop.size, (255, 255, 255)), mask)
    width = max(1, round(crop.width * HEIGHT / crop.height))
    return crop.resize((width, HEIGHT), Image.Resampling.LANCZOS)


def scan_score(image: str, scores: Sequence[float]) -> ScanScore:
    """The lines' scores of one scan gathered."""
    if not scores:
        return ScanScore(image=image, lines=0, mean=None, low=None, worst=None)
    return ScanScore(
        image=image,
        lines=len(scores),
        mean=sum(scores) / len(scores),
        low=sum(score < MATCH for score in scores) / len(scores),
        worst=min(scores),
    )


def match_score(no_match: float, match: float) -> float:
    """Jev's two logits as the probability of a match."""
    return 1 / (1 + math.exp(no_match - match))


def _prompts(pair: tuple[str, str]) -> tuple[str, list[dict[str, object]] | None]:
    """A scan's lines as Jev's prompts (crop, the label's text drawn at the crop's size); None for an unreadable scan
    or label file."""
    image, labels = pair
    try:
        lines = alto_lines(Path(labels))
        with Image.open(image) as scan:
            scan.load()
            crops = [(line, line_crop(scan, line)) for line in lines]
    except (OSError, ValueError, ET.ParseError):
        return image, None
    return image, [
        {
            "prompt": model_prompt(line.text),
            "multi_modal_data": {"image": [crop, render_text(line.text, FONT, crop.size)]},
        }
        for line, crop in crops
    ]


def _chunks(pairs: Sequence[tuple[str, str]], size: int) -> Iterator[Sequence[tuple[str, str]]]:
    for start in range(0, len(pairs), size):
        yield pairs[start : start + size]


def held(out: Path) -> set[str]:
    """The scans the parts of ``out`` hold."""
    return {
        json.loads(line)["image"] for part in out.glob("part-*.jsonl") for line in part.read_text("utf-8").splitlines()
    }


def main(
    pairs_file: Annotated[Path, typer.Argument(help="scan image and ALTO file a line, tab-separated")],
    out: Annotated[Path, typer.Argument(help="folder for the JSON-lines parts")],
    model: Annotated[
        str, typer.Option(help="Jev's checkpoint or repository")
    ] = "runs/htr/final/hub/diffu-minicpm-ocr-htr-jev",
    limit: Annotated[int | None, typer.Option(help="at most so many scans (a trial)")] = None,
    workers: Annotated[int, typer.Option(help="processes cutting and drawing lines")] = 16,
    memory: Annotated[float, typer.Option(help="the share of the GPU's memory vLLM may take")] = 0.3,
) -> None:
    """Score every scan of PAIRS_FILE that OUT does not hold yet."""
    from vllm import LLM, PoolingParams  # heavy, and only this entry point needs it

    out.mkdir(parents=True, exist_ok=True)
    pairs = [tuple(line.split("\t")) for line in pairs_file.read_text("utf-8").splitlines() if line.strip()]
    done = held(out)
    todo = [(image, labels) for image, labels in pairs if image not in done][:limit]
    print(f"{len(pairs)} scans; {len(done)} scored before; {len(todo)} to score", flush=True)
    jev = LLM(
        model=model,
        runner="pooling",
        convert="classify",
        hf_overrides={"num_labels": 2},
        limit_mm_per_prompt={"image": 2},
        gpu_memory_utilization=memory,
    )
    rows: list[ScanScore] = []
    scored = lost = lines = 0
    start = time.time()

    def flush() -> None:
        part = out / f"part-{time.time_ns()}.jsonl"
        part.with_suffix(".unfinished").write_text("".join(row.model_dump_json() + "\n" for row in rows), "utf-8")
        part.with_suffix(".unfinished").replace(part)
        rows.clear()

    with ProcessPoolExecutor(workers) as pool:
        for chunk in _chunks(todo, SCANS_A_BATCH):
            made = list(pool.map(_prompts, chunk))
            prompts = [prompt for _, scan_prompts in made for prompt in scan_prompts or []]
            outputs = (
                jev.classify(prompts, pooling_params=PoolingParams(use_activation=False), use_tqdm=False)
                if prompts
                else []
            )
            scores = iter(match_score(*output.outputs.probs[:2]) for output in outputs)
            for image, scan_prompts in made:
                if scan_prompts is None:
                    lost += 1
                    continue
                rows.append(scan_score(image, [next(scores) for _ in scan_prompts]))
            scored += len(chunk)
            lines += len(prompts)
            if len(rows) >= PART:
                flush()
            if scored % (SCANS_A_BATCH * 10) == 0:
                rate = scored / (time.time() - start)
                print(
                    f"{scored} of {len(todo)} scans, {lines} lines, {rate:.1f} scans a second, {lost} unreadable",
                    flush=True,
                )
    if rows:
        flush()
    print(f"done: {scored - lost} scans scored, {lost} unreadable, {lines} lines", flush=True)


if __name__ == "__main__":
    typer.run(main)
    sys.exit(0)
