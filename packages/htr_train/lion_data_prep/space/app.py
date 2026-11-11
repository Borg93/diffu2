"""Reading Swedish handwriting: the Swedish Lion TrOCR against our MiniCPM-OCR reader, alone and with its DSpark
draft, timed on the same region and replayed side by side; and Jev, the reader's proof-reader, scoring each line.

Everything runs in PyTorch on ZeroGPU (vLLM, where the draft normally runs, cannot run here): the reader's passes and
the draft's proposals are replayed from CUDA graphs (dspark_hf.py), for the reader alone and with the draft alike.
The readers run one after the other; each notes when every token became known, and the page replays them at that pace
(race_view.py). Nothing is sent to the page inside a timed stretch.
"""

# ZeroGPU patches CUDA when `spaces` is imported, so it comes before torch and anything that imports torch.
import spaces  # isort: skip

import base64
import io
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from PIL import Image
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

READER = "Gabriel/diffu-minicpm-ocr-htr"
DRAFT = "Gabriel/diffu-minicpm-ocr-htr-dspark"
JEV = "Gabriel/diffu-minicpm-ocr-htr-jev"
# a private copy of Riksarkivet's Swedish Lion Libre 2026 (large, character-based)
TROCR = os.environ.get("TROCR_REPO", "Gabriel/swedish-lion-trocr-large-handwritten-char")
HERE = Path(__file__).parent
FONT = HERE / "DejaVuSans.ttf"
EXAMPLES: list[dict[str, Any]] = json.loads((HERE / "examples" / "examples.json").read_text(encoding="utf-8"))
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
trocr_processor = TrOCRProcessor.from_pretrained(TROCR)
trocr = VisionEncoderDecoderModel.from_pretrained(TROCR).to("cuda").eval()  # float32, as Riksarkivet runs it

_engines: dict[int, D.FastDSpark] = {}  # per GPU worker process: CUDA graphs belong to the process that captured them
_lock = threading.Lock()
gr.set_static_paths([HERE / "examples"])  # the line crops in Jev's rows


class TrocrLine(BaseModel):
    """A line's reading and how many of its characters each decoding step had reached."""

    text: str
    ends: list[int]


class TrocrLane(BaseModel):
    """TrOCR's reading of a region's lines in one batch: its seconds and when each decoding step ended."""

    seconds: float
    steps: list[float]
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


class JevRow(BaseModel):
    """One line: its crop, its transcription, and Jev's score of TrOCR's reading and of the reader's, each of the
    line on its own."""

    index: int
    image: str
    truth: str
    trocr: str
    trocr_score: float
    trocr_exact: bool
    minicpm: str
    minicpm_score: float
    minicpm_exact: bool


class Race(BaseModel):
    """What RaceView draws: the region with its lines' outlines (fractions of the image) and each line's cumulative
    share of the transcription, the readers' lanes as they finish, then Jev's rows."""

    run: str = Field(default_factory=lambda: uuid4().hex[:12])
    image: str | None = None
    aspect: float = 1.0
    outlines: list[list[tuple[float, float]]] = Field(default_factory=list)
    shares: list[float] = Field(default_factory=list)
    status: str = ""
    done: bool = False
    trocr: TrocrLane | None = None
    plain: ReaderLane | None = None
    dspark: ReaderLane | None = None
    summary: str = ""
    jev: list[JevRow] = Field(default_factory=list)
    jev_total: int = 0
    jev_note: str = ""


class StepClock(StoppingCriteria):
    """Notes when each decoding step ends; it never stops the generation itself."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        self.times.append(time.perf_counter())
        never = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return cast(torch.BoolTensor, never)


def _example(example_id: str) -> dict[str, Any]:
    return next(e for e in EXAMPLES if e["id"] == example_id)


def _cer(prediction: str, truth: str) -> float:
    prediction, truth = " ".join(prediction.split()), " ".join(truth.split())
    return Levenshtein.distance(prediction, truth) / max(1, len(truth))


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


def _trocr_read(lines: list[Image.Image], max_new_tokens: int = 128) -> TrocrLane:
    """TrOCR's greedy readings of a batch of lines, timed from the images to the text, as Riksarkivet's evaluation
    runs it: the Swedish Lion starts decoding from its config's token (its generation_config names another, which
    costs about 2 points of CER), and its 192 x 1024 inputs need the position encoding interpolated."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    clock = StepClock()
    pixels = trocr_processor(images=lines, return_tensors="pt").pixel_values.to("cuda")
    ids = trocr.generate(
        pixels,
        decoder_start_token_id=trocr.config.decoder_start_token_id,
        interpolate_pos_encoding=True,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        stopping_criteria=StoppingCriteriaList([clock]),
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    steps = [round(at - started, 4) for at in clock.times]
    tokenizer = trocr_processor.tokenizer
    read = []
    for row in ids[:, 1:].tolist():  # after the start token
        text = tokenizer.decode(row, skip_special_tokens=True).strip()
        ends, shown = [], 0
        for step in range(1, len(steps) + 1):
            if shown < len(text):
                partial = tokenizer.decode(row[:step], skip_special_tokens=True).strip()
                shown = max(shown, min(len(text), len(partial)))
            ends.append(shown)
        read.append(TrocrLine(text=text, ends=ends))
    return TrocrLane(seconds=seconds, steps=steps, lines=read)


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
        _trocr_read([warm.crop((0, 0, 200, 40))], max_new_tokens=4)
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
    plain, spec = race.plain, race.dspark
    if plain is None or spec is None:
        return ""
    against_trocr = f" and {race.trocr.seconds / spec.seconds:.1f}× faster than TrOCR" if race.trocr else ""
    same = (
        "It wrote exactly what the reader writes alone."
        if identical
        else "Its reading differs from the reader's alone at a near-tied token: the reader checks 16 tokens at once, "
        "and a batch of another size can tip a tie the other way (the reader alone does the same between batch sizes)."
    )
    return (
        f"With the draft, the reader needed {spec.passes} passes instead of {plain.passes} for {spec.tokens} tokens, "
        f"keeping {spec.acceptance or 0:.1f} tokens per check on average (at most 16): "
        f"{plain.seconds / spec.seconds:.1f}× faster than alone{against_trocr}. {same}"
    )


def _jev_note(race: Race, truth: list[str]) -> str:
    whole = " ".join(truth)
    trocr_cer = _cer(" ".join(row.trocr for row in race.jev), whole)
    minicpm_cer = _cer(" ".join(row.minicpm for row in race.jev), whole)
    picked = [row.minicpm if row.minicpm_score > row.trocr_score else row.trocr for row in race.jev]
    picked_cer = _cer(" ".join(picked), whole)
    return (
        f"CER over these lines, each read on its own: TrOCR {100 * trocr_cer:.1f} %, MiniCPM "
        f"{100 * minicpm_cer:.1f} %, the reading Jev scores higher on each line {100 * picked_cer:.1f} %."
    )


def _duration(example_id: str, upload: Image.Image | None, with_jev: bool) -> int:
    return 75 if upload is not None else 50 + (15 if with_jev else 0)


@spaces.GPU(duration=_duration)
def compare(example_id: str, upload: Image.Image | None, with_jev: bool) -> Iterator[Race]:
    """Read one region three ways, one after the other, timing each from the images to the text: TrOCR on its lines
    in one batch, the MiniCPM-OCR reader alone, and the reader with its DSpark draft (the last two on the same CUDA
    graphs). For an example region, the reader then reads each line on its own, and Jev scores that reading and
    TrOCR's. Yields the race so far between the timed stretches."""
    race = Race(status="Preparing the GPU worker: a fresh one captures its CUDA graphs first…")
    yield race
    with _lock:
        engine = _engine()
        if upload is not None:
            region, lines, truth, limit = upload.convert("RGB"), [], None, UPLOAD_TOKENS
        else:
            folder = HERE / "examples" / example_id
            truth = _example(example_id)["lines"]
            region = Image.open(folder / "region.png").convert("RGB")
            lines = [Image.open(folder / f"line_{i:02d}.jpg").convert("RGB") for i in range(len(truth))]
            limit = REGION_TOKENS
        whole = "\n".join(truth) if truth else None

        if lines:
            race.status = f"TrOCR is reading the region's {len(lines)} lines…"
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

        if with_jev and race.trocr is not None and truth is not None:
            race.jev_total = len(lines)
            race.status = "The reader reads each line on its own, and Jev scores its reading and TrOCR's…"
            yield race
            for i, line in enumerate(lines):
                crop = _jev_crop(line)
                guess = race.trocr.lines[i].text
                own = _text(engine.decode(_region_inputs(line), LINE_TOKENS, EOS).tokens)
                row = JevRow(
                    index=i,
                    image=f"/gradio_api/file={HERE / 'examples' / example_id / f'line_{i:02d}.jpg'}",
                    truth=truth[i],
                    trocr=guess,
                    trocr_score=_jev_score(crop, guess),
                    trocr_exact=_same(guess, truth[i]),
                    minicpm=own,
                    minicpm_score=_jev_score(crop, own),
                    minicpm_exact=_same(own, truth[i]),
                )
                race.jev.append(row)
                yield race
            race.jev_note = _jev_note(race, truth)
        race.status, race.done = "", True
        yield race


def _stage(example_id: str, upload: Image.Image | None) -> Race:
    """The race before anyone has read: the region's image and, for an example, its lines' outlines."""
    if upload is not None:
        shown = upload.convert("RGB")
        shown.thumbnail((1000, 1000))
        buffer = io.BytesIO()
        shown.save(buffer, format="JPEG", quality=85)
        image = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        return Race(image=image, aspect=shown.width / shown.height)
    example = _example(example_id)
    width, height = example["region_size"]
    lengths = [max(1, len(line)) for line in example["lines"]]
    shares = [sum(lengths[: i + 1]) / sum(lengths) for i in range(len(lengths))]
    return Race(
        image=f"/gradio_api/file={HERE / 'examples' / example_id / 'region.png'}",
        aspect=width / height,
        outlines=example["outlines"],
        shares=shares,
    )


def run(example_label: str, upload: Image.Image | None, with_jev: bool) -> Iterator[dict[str, Any]]:
    """Read the chosen example region (or an uploaded one) with TrOCR, the MiniCPM-OCR reader and the reader with its
    DSpark draft, then let Jev score each line. Streams the race as each reader finishes: every reading with the time
    each token became known, the summary, and Jev's rows."""
    example_id = LABELS[example_label]
    stage = _stage(example_id, upload)
    stage.status = "Waiting for a GPU…"
    yield stage.model_dump()
    keep = {name: getattr(stage, name) for name in ("run", "image", "aspect", "outlines", "shares")}
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


def _label(example: dict[str, Any]) -> str:
    collection = example["collection"].removeprefix("GT-").rsplit("_", 1)[0].replace("_", " ")
    return f"{collection} · {len(example['lines'])} lines"


LABELS = {_label(e): e["id"] for e in EXAMPLES}

with gr.Blocks(title="Swedish HTR: TrOCR, MiniCPM-OCR and DSpark") as demo:
    gr.Markdown(
        "# Reading Swedish handwriting faster\n"
        "The same text region from an archive volume, read three ways on the same GPU. Riksarkivet's **Swedish Lion "
        "TrOCR** cannot read a region, so it gets the region as its text lines, cut out beforehand, and reads them as "
        f"one batch. Our **[MiniCPM-OCR reader](https://huggingface.co/{READER})** reads the region as one image, "
        f"alone and with its **[DSpark draft](https://huggingface.co/{DRAFT})**, which guesses 15 tokens at a time "
        f"for the reader to check in one pass. Then **[Jev](https://huggingface.co/{JEV})**, the reader's "
        "proof-reader, scores both readers' reading of each line. The regions are from volumes none of the models "
        "trained on."
    )
    with gr.Tab("Compare"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                choice = gr.Dropdown(list(LABELS), value=next(iter(LABELS)), label="Example region")
                with_jev = gr.Checkbox(value=True, label="Score each line with Jev")
                go = gr.Button("Read", variant="primary")
                with gr.Accordion("Or read your own region", open=False):
                    upload = gr.Image(type="pil", label="A region's crop")
                    gr.Markdown("Only the reader reads an upload: TrOCR and Jev need the region's lines.")
            with gr.Column(scale=2):
                preview = gr.Image(
                    value=str(HERE / "examples" / EXAMPLES[0]["id"] / "region.png"),
                    label="Region",
                    interactive=False,
                    height=330,
                )
        race_view = RaceView()
        choice.change(
            lambda label: str(HERE / "examples" / LABELS[label] / "region.png"),
            choice,
            preview,
            api_visibility="private",
        )
        go.click(run, [choice, upload, with_jev], race_view, api_name="compare", show_progress="minimal")
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
        first = max(EXAMPLES, key=lambda example: len(example["lines"]) == 17)  # a clear hand
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

demo.queue(default_concurrency_limit=2).launch(show_error=True, ssr_mode=False)
