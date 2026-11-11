"""Reading Swedish handwriting: the Swedish Lion TrOCR against our MiniCPM-OCR reader, alone and with its DSpark
draft, timed on the same region and replayed side by side; and Jev, the reader's proof-reader, scoring each line.

Everything runs in PyTorch on ZeroGPU (vLLM, where the draft normally runs, cannot run here): the reader's passes and
the draft's proposals are replayed from CUDA graphs (dspark_hf.py), for the reader alone and with the draft alike.
The readers run one after the other, one stream each: TrOCR reads the region's lines one by one (and once more as
one batch, shown beside), the reader reads the region as one sequence. Each notes when every token became known, and
once all are timed the page replays them from the same start at that pace (race_view.py). Nothing is sent to the page
inside a timed stretch. Under the live timings the page shows the same region's on our server, where vLLM serves the
reader (reference.py), and what the readers do there when everything is batched.
"""

# ZeroGPU patches CUDA when `spaces` is imported, so it comes before torch and anything that imports torch.
import spaces  # isort: skip

import base64
import filecmp
import io
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from rapidfuzz.distance import Levenshtein
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen3_5ForSequenceClassification,
    StoppingCriteria,
    StoppingCriteriaList,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

import dspark_hf as D
from jev_prompt import model_prompt
from jev_render import render_text, renderer_version
from race_view import RaceView
from reference import Reference, RegionTimes

READER = "Gabriel/diffu-minicpm-ocr-htr"
DRAFT = "Gabriel/diffu-minicpm-ocr-htr-dspark"
JEV = "Gabriel/diffu-minicpm-ocr-htr-jev"
# a private copy of Riksarkivet's Swedish Lion Libre 2026 (large, character-based)
TROCR = os.environ.get("TROCR_REPO", "Gabriel/swedish-lion-trocr-large-handwritten-char")
HERE = Path(__file__).parent
FONT = HERE / "DejaVuSans.ttf"
EXAMPLES: list[dict[str, Any]] = json.loads((HERE / "examples" / "examples.json").read_text(encoding="utf-8"))
REFERENCE = Reference.model_validate_json((HERE / "reference.json").read_text(encoding="utf-8"))
MAX_LEN = 4096  # the reader's cache: a region's prompt (up to about 2,600 tokens) and its reading; shorter is faster
MINICPM_IMAGES = {"downsample_mode": "4x", "max_slice_nums": 9}  # OpenBMB's settings, as the reader was trained
# new tokens at most: an example region, an upload (it must fit its GPU slot), a line
REGION_TOKENS, UPLOAD_TOKENS, LINE_TOKENS = 4096, 1500, 512
JEV_HEIGHT, JEV_JPEG = 48, 95  # Jev's line crops, as its training pairs were cut

# On CUDA at module level, as ZeroGPU asks: the weights move to the GPU when a GPU call starts.
processor = AutoProcessor.from_pretrained(READER)
reader = AutoModelForImageTextToText.from_pretrained(READER, dtype=torch.bfloat16).to("cuda").eval()
draft = D.DSparkDraft.from_folder(snapshot_download(DRAFT), reader.get_input_embeddings())
draft = draft.to("cuda", torch.bfloat16).eval()
_eos = reader.generation_config.eos_token_id
EOS: set[int] = set(_eos) if isinstance(_eos, list) else {int(_eos)}
jev_processor = AutoProcessor.from_pretrained(JEV)
jev = Qwen3_5ForSequenceClassification.from_pretrained(JEV, num_labels=2, dtype=torch.bfloat16).to("cuda").eval()
try:
    trocr_processor = TrOCRProcessor.from_pretrained(TROCR)
    trocr = VisionEncoderDecoderModel.from_pretrained(TROCR).to("cuda").eval()  # float32, as Riksarkivet runs it
except (OSError, HfHubHTTPError) as error:  # its weights are private: without a token the app runs without it
    print(f"TrOCR is not loaded ({type(error).__name__}): the app runs without it", flush=True)
    trocr_processor = trocr = None
NO_TROCR = (
    "The Swedish Lion TrOCR is not available here: its weights are private, and this Space has no HF_TOKEN secret "
    "that can read them. The reader and its draft run without it; Jev's line-by-line comparison needs both readers."
)

_engines: dict[int, D.FastDSpark] = {}  # per GPU worker process: CUDA graphs belong to the process that captured them
_lock = threading.Lock()
gr.set_static_paths([HERE / "examples"])  # the line crops in Jev's rows


class TrocrLine(BaseModel):
    """A line's reading, when each of its decoding steps ended (seconds from the start of the region's first line) and
    how many of its characters each step had reached."""

    text: str
    steps: list[float]
    ends: list[int]


class TrocrLane(BaseModel):
    """TrOCR's reading of a region's lines, one line after the other: its seconds and each line's steps; and the
    seconds the same lines took as one batch."""

    seconds: float
    batch_seconds: float
    lines: list[TrocrLine]
    cer: float | None = None

    @property
    def texts(self) -> list[str]:
        return [line.text for line in self.lines]


class ReaderLane(BaseModel):
    """The reader's reading of a region: its seconds and passes, and each token's text with when it became known and
    whether the draft proposed it."""

    seconds: float
    tokens: int
    passes: int
    pieces: list[tuple[float, str, bool]]
    acceptance: float | None = None
    cer: float | None = None

    @property
    def text(self) -> str:
        return "".join(piece for _, piece, _ in self.pieces).strip()


class ServerTimes(RegionTimes):
    """A region's timings on our server, with its GPU, the day they were measured and the runs they are medians of."""

    gpu: str
    measured: str
    runs: int


class Rate(BaseModel):
    """What a reader reads a second when everything is batched, and how it was batched."""

    lane: Literal["trocr", "plain", "dspark"]
    name: str
    how: str
    per_second: float


class JevReading(BaseModel):
    """One reader's reading of a line: its text; the text in stretches, each with whether it agrees with the ground
    truth; its errors (characters added, dropped or changed against the ground truth); Jev's score (how sure Jev is,
    0 to 1, that the reading is exactly the line); and whether it is exactly the ground truth."""

    text: str
    marks: list[tuple[str, bool]]
    errors: int
    score: float
    exact: bool


class JevRow(BaseModel):
    """One line: its crop, its ground truth, TrOCR's reading and the reader's (each of the line on its own) with Jev's
    scores; the reading Jev picks ("" where both read the same) and the one with fewer errors ("" where they have as
    many)."""

    index: int
    image: str
    truth: str
    trocr: JevReading
    minicpm: JevReading
    kept: Literal["trocr", "minicpm", ""]
    better: Literal["trocr", "minicpm", ""]


class JevTally(BaseModel):
    """What Jev did on a region's lines: where the two readings differ, where one of them has fewer errors and where
    Jev picked that one; and the CER of each reader, of Jev's picks, and of the reading with fewer errors on every
    line (the best possible picks)."""

    lines: int
    differ: int
    one_better: int
    kept_better: int
    trocr: float
    minicpm: float
    kept: float
    best: float


class Race(BaseModel):
    """What RaceView draws: the region with its lines' outlines (fractions of the image) and each line's cumulative
    share of the transcription, the readers' lanes as they are timed (``ready`` once all are: the replay starts), the
    same region's timings on our server and the batched rates, then Jev's rows."""

    run: str = Field(default_factory=lambda: uuid4().hex[:12])
    image: str | None = None
    aspect: float = 1.0
    outlines: list[list[tuple[float, float]]] = Field(default_factory=list)
    shares: list[float] = Field(default_factory=list)
    status: str = ""
    notice: str = ""
    ready: bool = False
    done: bool = False
    trocr: TrocrLane | None = None
    plain: ReaderLane | None = None
    dspark: ReaderLane | None = None
    summary: str = ""
    server: ServerTimes | None = None
    server_said: str = ""
    batched: list[Rate] = Field(default_factory=list)
    batched_note: str = ""
    jev: list[JevRow] = Field(default_factory=list)
    jev_total: int = 0
    jev_tally: JevTally | None = None


# Everything batched, on our server: the first 1,000 test lines, a whole GPU for TrOCR, the two vLLM servers sharing
# one and asked in turn, every image new to them (docs/HTR_RUNS.md: bench_three.sh on 9 Nov, bench_cold.sh on 11 Nov)
BATCHED = [
    Rate(lane="trocr", name="Swedish Lion TrOCR", how="batches of 32 lines", per_second=6.54),
    Rate(lane="plain", name="Reader alone, in vLLM", how="64 requests at a time", per_second=67.26),
    Rate(lane="dspark", name="Reader + DSpark, in vLLM", how="64 requests at a time", per_second=77.15),
]
# This Space's decoding against vLLM on that server's GPU, the six example regions (docs/HTR_RUNS.md, 11 Nov 2026)
SAME_GPU = (
    "On such a whole GPU this Space's PyTorch decoding was as fast as vLLM with the draft and 1.2–1.8× slower alone "
    "(six other regions); the rest of the gap to the live bars is the half GPU."
)
BATCHED_NOTE = (
    "The first 1,000 test lines on one RTX PRO 6000 Blackwell, every image new to the server. One at a time on the "
    "same lines: TrOCR 4.3 lines a second, the reader 13.3, with its draft 18.2. Whole regions, 64 at a time: 11.9 a "
    "second with the reader alone, 12.7 with its draft."
)


class StepClock(StoppingCriteria):
    """Notes when each decoding step ends; it never stops the generation itself."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self._never: torch.Tensor | None = None  # made once: a tensor a step would cost the reader it times

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        self.times.append(time.perf_counter())
        if self._never is None:
            self._never = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return cast(torch.BoolTensor, self._never)


def _example(example_id: str) -> dict[str, Any]:
    return next(e for e in EXAMPLES if e["id"] == example_id)


def _errors(prediction: str, truth: str) -> int:
    """Characters added, dropped or changed against the ground truth, runs of spaces counted as one."""
    return Levenshtein.distance(" ".join(prediction.split()), " ".join(truth.split()))


def _cer(prediction: str, truth: str) -> float:
    return _errors(prediction, truth) / max(1, len(" ".join(truth.split())))


def _same(prediction: str, truth: str) -> bool:
    return " ".join(prediction.split()) == " ".join(truth.split())


def _region_inputs(image: Image.Image) -> dict[str, Any]:
    content = [{"type": "image", "image": image}, {"type": "text", "text": "Text Recognition:"}]
    batch = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        processor_kwargs=MINICPM_IMAGES,
    )
    inputs = dict(batch.to("cuda"))
    inputs["downsample_mode"] = MINICPM_IMAGES["downsample_mode"]
    return inputs


def _trocr_ids(lines: list[Image.Image], clock: StepClock | None = None, max_new_tokens: int = 128) -> torch.Tensor:
    """TrOCR's greedy token ids for a batch of lines, as Riksarkivet's evaluation runs it: the Swedish Lion starts
    decoding from its config's token (its generation_config names another, which costs about 2 points of CER), and
    its 192 x 1024 inputs need the position encoding interpolated."""
    model, images = trocr, trocr_processor
    if model is None or images is None:
        raise RuntimeError("TrOCR is not loaded")
    pixels = images(images=lines, return_tensors="pt").pixel_values.to("cuda")
    return model.generate(
        pixels,
        decoder_start_token_id=model.config.decoder_start_token_id,
        interpolate_pos_encoding=True,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        stopping_criteria=StoppingCriteriaList([clock] if clock else []),
    )


def _trocr_read(lines: list[Image.Image]) -> TrocrLane:
    """TrOCR's readings of a region's lines, one line after the other, timed from the images to the token ids; then
    the seconds the same lines take as one batch."""
    images = trocr_processor
    if images is None:
        raise RuntimeError("TrOCR is not loaded")
    torch.cuda.synchronize()
    started = time.perf_counter()
    rows = []
    for line in lines:
        clock = StepClock()
        ids = _trocr_ids([line], clock)[0, 1:].tolist()  # after the start token; the copy waits for the GPU
        rows.append((ids, [round(at - started, 4) for at in clock.times]))
    seconds = time.perf_counter() - started
    _, batch_seconds = _timed(lambda: _trocr_ids(lines))
    tokenizer = images.tokenizer
    read = []
    for ids, steps in rows:
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        ends, shown = [], 0
        for step in range(1, len(steps) + 1):
            if shown < len(text):
                partial = tokenizer.decode(ids[:step], skip_special_tokens=True).strip()
                shown = max(shown, min(len(text), len(partial)))
            ends.append(shown)
        read.append(TrocrLine(text=text, steps=steps, ends=ends))
    return TrocrLane(seconds=seconds, batch_seconds=batch_seconds, lines=read)


def _reader_lane(decoded: D.Decoded, seconds: float) -> ReaderLane:
    """A reading as a lane: each token's share of the text (nothing while a character is still incomplete), timed from
    the start of the whole stretch, preprocessing included."""
    tokenizer = processor.tokenizer
    before_decoding = seconds - decoded.seconds
    pieces, prefix, read = [], 0, 0
    for i, (at, from_draft) in enumerate(zip(decoded.times, decoded.from_draft, strict=True)):
        known = tokenizer.decode(decoded.tokens[prefix:read], skip_special_tokens=True)
        longer = tokenizer.decode(decoded.tokens[prefix : i + 1], skip_special_tokens=True)
        piece = ""
        if len(longer) > len(known) and not longer.endswith("�"):
            piece, prefix, read = longer[len(known) :], read, i + 1
        pieces.append((round(before_decoding + at, 4), piece, from_draft))
    return ReaderLane(
        seconds=seconds,
        tokens=len(decoded.tokens),
        passes=decoded.reader_passes,
        pieces=pieces,
        acceptance=decoded.acceptance_length if decoded.cycles else None,
    )


def _text(tokens: list[int]) -> str:
    return processor.decode(tokens, skip_special_tokens=True).strip()


def _jev_crop(line: Image.Image) -> Image.Image:
    """A line's crop as Jev was trained on it: 48 pixels high, its aspect kept, through JPEG at quality 95."""
    image = line.convert("RGB")
    if image.height != JEV_HEIGHT:
        width = max(1, round(image.width * JEV_HEIGHT / image.height))
        image = image.resize((width, JEV_HEIGHT), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JEV_JPEG)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def _jev_score(crop: Image.Image, text: str) -> float:
    rendering = render_text(text, FONT, crop.size)
    inputs = jev_processor(text=[model_prompt(text)], images=[crop, rendering], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = jev(**inputs).logits[0].float()
    return float(torch.softmax(logits, -1)[1])


def _engine() -> D.FastDSpark:
    """This worker's reader and draft on CUDA graphs, captured and warmed on first use (no timing includes it)."""
    engine = _engines.get(os.getpid())
    if engine is None:
        engine = D.FastDSpark(reader, draft, MAX_LEN)
        warm = Image.open(HERE / "examples" / EXAMPLES[0]["id"] / "region.png").convert("RGB")
        engine.warm(_region_inputs(warm), EOS)
        if trocr is not None:
            _trocr_ids([warm.crop((0, 0, 200, 40))], max_new_tokens=4)
        _jev_score(_jev_crop(warm.crop((0, 0, 200, 40))), "warm")
        _engines[os.getpid()] = engine
    return engine


def _timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def _summary(race: Race, identical: bool) -> str:
    plain, spec, lines = race.plain, race.dspark, race.trocr
    if plain is None or spec is None:
        return ""
    said = [
        f"One stream each, nothing batched: with its draft the reader read the region in {spec.seconds:.2f} s, "
        f"{plain.seconds / spec.seconds:.1f}× faster than alone"
    ]
    if lines is not None:
        said[0] += (
            f" and {lines.seconds / spec.seconds:.1f}× faster than TrOCR reading the {len(lines.lines)} lines one "
            "after the other"
        )
        said.append(
            f"As one batch the lines decode side by side, and TrOCR took {lines.batch_seconds:.2f} s "
            f"({lines.batch_seconds / spec.seconds:.1f}× the reader with its draft); a region is one sequence, so "
            "the reader has nothing to batch here. It batches whole regions, in vLLM"
        )
    said.append(
        f"The draft: {spec.passes} reader passes instead of {plain.passes} for {spec.tokens} tokens, "
        f"{spec.acceptance or 0:.1f} tokens kept per check on average (at most 16)"
    )
    said.append(
        "It wrote exactly what the reader writes alone"
        if identical
        else "Its reading differs from the reader's alone at a near-tied token: the reader checks 16 tokens at once, "
        "and a batch of another size can tip a tie the other way (the reader alone does the same between batch sizes)"
    )
    return ". ".join(said) + "."


def _marks(reading: str, truth: str) -> list[tuple[str, bool]]:
    """A reading in stretches, each with whether it agrees with the ground truth; where the reading lacks characters,
    a caret stands for them."""
    reading, truth = " ".join(reading.split()), " ".join(truth.split())
    marks = []
    for step in Levenshtein.opcodes(reading, truth):
        if step.tag == "insert":
            marks.append(("‸", False))
        else:
            marks.append((reading[step.src_start : step.src_end], step.tag == "equal"))
    return marks


def _jev_reading(crop: Image.Image, reading: str, truth: str) -> JevReading:
    return JevReading(
        text=reading,
        marks=_marks(reading, truth),
        errors=_errors(reading, truth),
        score=_jev_score(crop, reading),
        exact=_same(reading, truth),
    )


def _jev_row(index: int, image: str, truth: str, crop: Image.Image, guess: str, own: str) -> JevRow:
    """A line's row: Jev's score of TrOCR's reading and of the reader's, the one it picks and the one with fewer
    errors."""
    first = _jev_reading(crop, guess, truth)
    if _same(guess, own):
        return JevRow(index=index, image=image, truth=truth, trocr=first, minicpm=first, kept="", better="")
    second = _jev_reading(crop, own, truth)
    kept = "minicpm" if second.score > first.score else "trocr"
    better = "" if first.errors == second.errors else ("trocr" if first.errors < second.errors else "minicpm")
    return JevRow(index=index, image=image, truth=truth, trocr=first, minicpm=second, kept=kept, better=better)


def _jev_tally(rows: list[JevRow]) -> JevTally:
    """Jev over a region's lines: the CERs are over the region's text, each line read on its own."""
    whole = " ".join(row.truth for row in rows)
    chosen = [row for row in rows if row.kept]
    better = [row for row in chosen if row.better]

    def cer(pick: Callable[[JevRow], JevReading]) -> float:
        return _cer(" ".join(pick(row).text for row in rows), whole)

    return JevTally(
        lines=len(rows),
        differ=len(chosen),
        one_better=len(better),
        kept_better=sum(row.kept == row.better for row in better),
        trocr=cer(lambda row: row.trocr),
        minicpm=cer(lambda row: row.minicpm),
        kept=cer(lambda row: row.minicpm if row.kept == "minicpm" else row.trocr),
        best=cer(lambda row: row.minicpm if row.better == "minicpm" else row.trocr),
    )


def _duration(example_id: str | None, upload: Image.Image | None, with_jev: bool) -> int:
    return 75 if upload is not None else 55 + (15 if with_jev else 0)


@spaces.GPU(duration=_duration)
def compare(example_id: str | None, upload: Image.Image | None, with_jev: bool) -> Iterator[Race]:
    """Read one region with one stream each, one reader after the other, timing each from the images to the text:
    TrOCR on its lines one by one (and once more as one batch), the MiniCPM-OCR reader alone, and the reader with its
    DSpark draft (the last two on the same CUDA graphs). For an example region, the reader then reads each line on
    its own, and Jev scores that reading and TrOCR's. Yields the race so far between the timed stretches."""
    race = Race(status="Preparing the GPU worker: a fresh one captures its CUDA graphs first…")
    yield race
    with _lock:
        engine = _engine()
        if upload is not None or example_id is None:
            region, lines, truth, limit = cast(Image.Image, upload), [], None, UPLOAD_TOKENS
        else:
            folder = HERE / "examples" / example_id
            truth = _example(example_id)["lines"]
            region = Image.open(folder / "region.png").convert("RGB")
            lines = [Image.open(folder / f"line_{i:02d}.jpg").convert("RGB") for i in range(len(truth))]
            limit = REGION_TOKENS
        whole = "\n".join(truth) if truth else None

        if trocr is None:
            race.notice = NO_TROCR
        elif lines:
            race.status = f"TrOCR is reading the region's {len(lines)} lines, one after the other, then as one batch…"
            yield race
            race.trocr = _trocr_read(lines)
            race.trocr.cer = _cer("\n".join(race.trocr.texts), whole) if whole else None

        race.status = "The MiniCPM-OCR reader is reading the region, one token per pass…"
        yield race
        plain, seconds = _timed(lambda: D.greedy(engine.reader, _region_inputs(region), limit, EOS))
        race.plain = _reader_lane(plain, seconds)
        race.plain.cer = _cer(race.plain.text, whole) if whole else None

        race.status = "The reader is reading it again, now with its DSpark draft…"
        yield race
        spec, seconds = _timed(lambda: engine.decode(_region_inputs(region), limit, EOS))
        race.dspark = _reader_lane(spec, seconds)
        race.dspark.cer = _cer(race.dspark.text, whole) if whole else None
        race.summary = _summary(race, identical=plain.tokens == spec.tokens)
        race.ready = True

        if with_jev and race.trocr is not None and truth is not None and example_id is not None:
            race.jev_total = len(lines)
            race.status = "The reader reads each line on its own, and Jev scores its reading and TrOCR's…"
            yield race
            for i, line in enumerate(lines):
                own = _text(engine.decode(_region_inputs(line), LINE_TOKENS, EOS).tokens)
                image = f"/gradio_api/file={HERE / 'examples' / example_id / f'line_{i:02d}.jpg'}"
                race.jev.append(_jev_row(i, image, truth[i], _jev_crop(line), race.trocr.lines[i].text, own))
                yield race
            race.jev_tally = _jev_tally(race.jev)
        race.status, race.done = "", True
        yield race


def _stage(example_id: str | None, upload: Image.Image | None) -> Race:
    """The race before anyone has read: the region's image and, for an example, its lines' outlines and its timings
    on our server."""
    if upload is not None or example_id is None:
        shown = cast(Image.Image, upload).copy()
        shown.thumbnail((1000, 1000))
        buffer = io.BytesIO()
        shown.save(buffer, format="JPEG", quality=85)
        image = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        return Race(image=image, aspect=shown.width / shown.height, batched=BATCHED, batched_note=BATCHED_NOTE)
    example = _example(example_id)
    width, height = example["region_size"]
    lengths = [max(1, len(line)) for line in example["lines"]]
    shares = [sum(lengths[: i + 1]) / sum(lengths) for i in range(len(lengths))]
    times = REFERENCE.regions.get(example_id)
    return Race(
        image=f"/gradio_api/file={HERE / 'examples' / example_id / 'region.png'}",
        aspect=width / height,
        outlines=example["outlines"],
        shares=shares,
        server=ServerTimes(**times.model_dump(), gpu=REFERENCE.gpu, measured=REFERENCE.measured, runs=REFERENCE.runs)
        if times
        else None,
        server_said=_server_said(times) if times else "",
        batched=BATCHED,
        batched_note=BATCHED_NOTE,
    )


def _server_said(times: RegionTimes) -> str:
    return (
        f"Served by vLLM, the reader read this region in {times.vllm:.2f} s alone and {times.vllm_dspark:.2f} s with "
        f"its draft: {times.trocr_single / times.vllm:.1f}× and {times.trocr_single / times.vllm_dspark:.1f}× faster "
        f"than TrOCR reading the {times.lines} lines one after the other, {times.trocr_batch / times.vllm:.1f}× and "
        f"{times.trocr_batch / times.vllm_dspark:.1f}× faster than TrOCR's one batch. {SAME_GPU}"
    )


def _example_of(image: str) -> str | None:
    """The example whose region is the image in the box, or None for a visitor's own image."""
    for example in EXAMPLES:
        if filecmp.cmp(image, HERE / "examples" / example["id"] / "region.png", shallow=False):
            return example["id"]
    return None


def run(region: str | None, with_jev: bool) -> Iterator[dict[str, Any]]:
    """Read the region in the box (an example, or a crop of your own) with one stream each: TrOCR on an example's
    lines, the MiniCPM-OCR reader and the reader with its DSpark draft; then let Jev score each line of an example.
    Streams the race as each reader is timed: every reading with the time each token became known, the summary, the
    same region's timings on our server, and Jev's rows."""
    if region is None:
        raise gr.Error("Pick an example region, or drop a region's crop on the box.")
    example_id = _example_of(region)
    upload = None if example_id else ImageOps.exif_transpose(Image.open(region)).convert("RGB")
    stage = _stage(example_id, upload)
    stage.status = "Waiting for a GPU…"
    yield stage.model_dump()
    keep = {
        name: getattr(stage, name)
        for name in ("run", "image", "aspect", "outlines", "shares", "server", "server_said", "batched", "batched_note")
    }
    for race in compare(example_id, upload, with_jev):
        yield race.model_copy(update=keep).model_dump()


@spaces.GPU(duration=30)
def score_line(line: Image.Image | None, text: str) -> tuple[dict[str, float], Image.Image]:
    """Jev's score (0 to 1) that a candidate text is exactly what a handwritten line says, and the rendering of the
    candidate it compares with the line."""
    if line is None or not text.strip():
        raise gr.Error("Give a line image and a candidate text.")
    with _lock:
        crop = _jev_crop(line)
        score = _jev_score(crop, text.strip())
    return {"matches": score, "does not match": 1 - score}, render_text(text.strip(), FONT, crop.size)


# the examples table's thumbnails, small enough for its six rows to stand beside the region box
EXAMPLES_CSS = (
    "#regions .container.table { width: 44px; height: 44px; } #regions td { padding-top: 4px; padding-bottom: 4px; }"
)


def _collection(example: dict[str, Any]) -> str:
    return example["collection"].removeprefix("GT-").rsplit("_", 1)[0].replace("_", " ")


def _region(example: dict[str, Any]) -> str:
    return str(HERE / "examples" / example["id"] / "region.png")


with gr.Blocks(title="Swedish HTR: TrOCR, MiniCPM-OCR and DSpark") as demo:
    gr.Markdown(
        "# Reading Swedish handwriting faster\n"
        "One text region from an archive volume, read on the same GPU with **one stream each and nothing batched**: "
        "Riksarkivet's **Swedish Lion TrOCR** on the region's text lines, cut out beforehand, one after the other; "
        f"our **[MiniCPM-OCR reader](https://huggingface.co/{READER})** on the region as one image, alone and with "
        f"its **[DSpark draft](https://huggingface.co/{DRAFT})**, which guesses 15 tokens at a time for the reader "
        f"to check in one pass. Then **[Jev](https://huggingface.co/{JEV})**, the reader's proof-reader, scores "
        "both readings of each line. Under the race: the same region on our server, where **vLLM** serves the "
        "reader, and both readers when everything is batched."
    )
    with gr.Tab("Compare"):
        with gr.Row():
            with gr.Column(scale=5, min_width=320):
                region = gr.Image(
                    value=_region(EXAMPLES[0]),
                    type="filepath",
                    image_mode=None,
                    sources=["upload", "clipboard"],
                    label="Region",
                    height=300,
                    buttons=["fullscreen"],
                )
                with gr.Row():
                    with_jev = gr.Checkbox(
                        value=True,
                        label="Score each line with Jev",
                        info="Examples only: TrOCR and Jev need a region's lines, so only the reader reads your own crop.",
                        scale=3,
                    )
                    go = gr.Button("Read", variant="primary", scale=2, min_width=140)
            with gr.Column(scale=4, min_width=320):
                # an example's own columns: filled by its row, shown nowhere else
                collection = gr.Textbox(label="Collection", visible=False)
                count = gr.Number(label="Lines", visible=False)
                gr.Examples(
                    [[_region(e), _collection(e), len(e["lines"])] for e in EXAMPLES],
                    [region, collection, count],
                    label="Examples drawn at random from test pages no model here trained on (or upload or paste your own)",
                    examples_per_page=len(EXAMPLES),
                    cache_examples=False,
                    elem_id="regions",
                )
        race_view = RaceView()
        go.click(run, [region, with_jev], race_view, api_name="compare", show_progress="minimal")
    with gr.Tab("Try Jev"):
        gr.Markdown(
            "Jev scores whether a text is exactly what a **single handwritten line** says. Upload a line crop and type "
            "a reading; change a letter and watch the score drop."
        )
        with gr.Row():
            with gr.Column():
                line_in = gr.Image(type="pil", label="Line image", height=120)
                text_in = gr.Textbox(label="Candidate text")
                score_go = gr.Button("Score", variant="primary")
            with gr.Column():
                score_out = gr.Label(label="Jev")
                render_out = gr.Image(label="The candidate rendered, as Jev compares it", height=120)
        first = EXAMPLES[0]
        gr.Examples(
            [[str(HERE / "examples" / first["id"] / f"line_{i:02d}.jpg"), first["lines"][i]] for i in range(3)],
            [line_in, text_in],
            cache_examples=False,
        )
        score_go.click(score_line, [line_in, text_in], [score_out, render_out], api_name="score_line")
    with gr.Tab("How it works"):
        gr.Markdown(
            (HERE / "HOW_IT_WORKS.md").read_text(encoding="utf-8").replace("@RENDERER@", renderer_version(FONT))
        )

demo.queue(default_concurrency_limit=2).launch(show_error=True, ssr_mode=False, css=EXAMPLES_CSS)
