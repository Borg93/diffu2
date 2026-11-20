"""diffu archive pages: a Hugging Face Space (Gradio, ZeroGPU) that draws a synthetic archive page from a layout of
boxes and a sample page, with diffu's fine-tuned MOSS-Image model (Gabriel/diffu-moss-above).

The page is laid out in an editor of our own (``pagedemo/editor.js``, a ``gr.HTML`` component): boxes drawn on the
page or loaded from a template, each a block of line slots with its text. It is planned on CPU exactly as the page
pipeline plans a page given as a layout (``pagedemo/planning.py``), drawn on the GPU (``pagedemo/drawing.py``), every
line read back by the pipeline's judge, and returned with its ALTO and PAGE labels.

``PAGEDEMO_STUB=1`` serves the page without loading the model (the editor and the planning only: a drawing is a blank
sheet), for work on the interface.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # noqa: E402, I001  # before torch: on ZeroGPU it patches torch's CUDA calls

import base64  # noqa: E402
import functools  # noqa: E402
import html  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from diffu_moss.modular_pipelines.moss.encoders import MAX_PROMPT_TOKENS, prompt_tokens  # noqa: E402
from diffu_moss.train.pairs import SampleSize  # noqa: E402
from diffu_page.draw.blocks.ground_truth import crop_labels  # noqa: E402
from diffu_page.draw.plan import CropPlan  # noqa: E402
from diffu_page.draw.sample import sized_sample  # noqa: E402
from diffu_page.output import load_rgb  # noqa: E402
from diffu_page.page.crops.geometry import LayoutBox  # noqa: E402
from diffu_page.page.crops.grid import LATENT_GRID  # noqa: E402
from diffu_page.page.labels import LabelFormat, export  # noqa: E402
from diffu_page.quality.judge import MAX_LINE_CER, LineVerdict  # noqa: E402
from pagedemo.document import Document  # noqa: E402
from pagedemo.drawing import DIT_REPO, DIT_REVISION, Drawn, Models, draw, load_models  # noqa: E402
from pagedemo.planning import PagePlan, layout_prompt_of, page_plan, planned  # noqa: E402
from pagedemo.record import DrawRecord  # noqa: E402
from pagedemo.templates import TEMPLATES  # noqa: E402

HERE = Path(__file__).parent
STUB = os.environ.get("PAGEDEMO_STUB") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = 2**31 - 1
STEPS, GUIDANCE, MEGAPIXELS = 28, 2.0, 2.0  # a page's defaults: the guidance the model card suggests, at its size
LONGEST_SAMPLE_SIDE = 4096  # px: a larger upload is shrunk to it once (the model reads a sample at 2 MP at most)
FAST, GUIDED = "fast", "guided"  # a page's mode: the distilled student's few unguided steps, or the DiT's guided ones
# The GPU seconds a page is asked for: the model's start on a GPU, its text encoder, VAE and judge, then each pass of
# the DiT by the tokens it reads (the page's and the sample's latent cells and the prompt's tokens; two passes a step
# under guidance). On ZeroGPU's half of an RTX PRO 6000 (20 Nov 2026, a letter of 13,900 tokens in 8 passes) the DiT
# alone took 0.16 s a pass and thousand tokens and with the student's LoRA beside it 0.18, the rest of a call under
# 10 s: asked for with a margin, as a call that outruns its seconds is cut off.
SETUP_SECONDS, LONGEST_SECONDS = 20, 300
SECONDS_PER_PASS_KILOTOKEN = {FAST: 0.21, GUIDED: 0.18}

MODELS: Models | None = None if STUB else load_models(DEVICE, student=HERE / "student")
STUDENT_STEPS = 8 if MODELS is None or MODELS.student is None else MODELS.student.student_steps
HAS_STUDENT = STUB or (MODELS is not None and MODELS.student is not None)


def passes(mode: str, steps: int, guidance: float) -> int:
    """The DiT passes a page takes: the student's steps, or the steps asked, twice under guidance."""
    if mode == FAST:
        return STUDENT_STEPS
    return steps * (1 if math.isclose(guidance, 1.0) else 2)


def dit_tokens(plan: CropPlan, sample: Image.Image, sample_lines: list[LayoutBox], prompt: str) -> int:
    """The tokens the DiT reads a pass for ``plan`` drawn from ``sample``: the page's and the sized sample's latent
    cells, and the prompt's tokens (an estimate from its length in the stub, which has no tokenizer)."""
    seen = sized_sample(sample, SampleSize.LEGIBLE, plan=plan, lines=sample_lines)
    cells = (plan.width * plan.height + seen.width * seen.height) // LATENT_GRID**2
    words = token_count(prompt)
    return cells + (len(prompt) // 3 if words is None else words)


def gpu_seconds(tokens: int, mode: str, steps: int, guidance: float) -> int:
    """The GPU seconds asked for a page whose DiT reads ``tokens`` tokens a pass, drawn in ``mode``."""
    drawing = passes(mode, steps, guidance) * tokens / 1000 * SECONDS_PER_PASS_KILOTOKEN[mode]
    return min(LONGEST_SECONDS, math.ceil(SETUP_SECONDS + drawing))


def _duration(
    plan: CropPlan,
    sample: Image.Image,
    sample_lines: list[LayoutBox],
    steps: int,
    guidance: float,
    seed: int,
    mode: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),  # noqa: B008  # as draw_on_gpu's own
) -> int:
    """The GPU seconds ZeroGPU is asked for a page: it calls this with the arguments :func:`draw_on_gpu` is called
    with, so the two share one signature."""
    return gpu_seconds(dit_tokens(plan, sample, sample_lines, layout_prompt_of(plan)), mode, steps, guidance)


@spaces.GPU(duration=_duration)
def draw_on_gpu(
    plan: CropPlan,
    sample: Image.Image,
    sample_lines: list[LayoutBox],
    steps: int,
    guidance: float,
    seed: int,
    mode: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),  # noqa: B008  # Gradio's own idiom
) -> Drawn:
    """A planned page drawn and read back on the GPU."""
    if MODELS is None:  # the stub: a blank sheet, at once
        blank = Image.new("RGB", (plan.width, plan.height), (241, 234, 216))
        return Drawn(image=blank, verdicts=None, read_error="no model is loaded", draw_seconds=0.0, read_seconds=0.0)
    return draw(MODELS, plan, sample, sample_lines, steps=steps, guidance=guidance, seed=seed, fast=mode == FAST)


def data_uri(image: Image.Image, side: int) -> str:
    """``image`` as a JPEG data URI no larger than ``side`` px, small enough to show in the editor."""
    shown = image.convert("RGB")
    shown.thumbnail((side, side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    shown.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def editor_value(document: Document, result: str | None) -> str:
    """The editor's value: the document, and the drawn page shown under its boxes (None: none)."""
    return json.dumps({**document.model_dump(mode="json"), "result": result}, ensure_ascii=False)


def chips(*parts: str, warn: bool = False, title: str = "") -> str:
    """A status line of small labels (``title``: what hovering over it explains)."""
    kind = "pd-chips pd-warn" if warn else "pd-chips"
    labels = "".join(f"<span>{html.escape(part)}</span>" for part in parts if part)
    return f"<div class='{kind}' title='{html.escape(title, quote=True)}'>{labels}</div>"


ASKED_TITLE = (
    "ZeroGPU reserves the seconds a page asks for out of your daily GPU quota before it is drawn; "
    "a page is usually drawn in about two thirds of them."
)


def token_count(prompt: str) -> int | None:
    """The tokens the text encoder reads for ``prompt``; None in the stub, which has no tokenizer."""
    return None if MODELS is None else prompt_tokens(MODELS.pipe.tokenizer, prompt)


@functools.lru_cache(maxsize=8)
def sample_image(path: str) -> Image.Image:
    """The sample page at ``path`` (an upload's own file, or an example's), shrunk if it is larger than the model can
    use. Read only: the cache hands one image to every request that draws from it."""
    image = load_rgb(Path(path))
    image.thumbnail((LONGEST_SAMPLE_SIDE, LONGEST_SAMPLE_SIDE), Image.Resampling.LANCZOS)
    return image


def status(
    value: str, sample: str | None, megapixels: float, steps: int, guidance: float, mode: str, sample_zoom: float
) -> tuple[str, str]:
    """What the page in the editor comes to before it is drawn: its lines, its prompt's length and the GPU time it
    will ask for as a status line, and the prompt itself."""
    try:
        document = Document.parse(value)
        image = sample_image(sample) if sample else None
        if image is None:
            plan, notes = page_plan(document, megapixels=megapixels)
            ready = PagePlan(plan=plan, prompt=layout_prompt_of(plan), sample_lines=[], notes=notes)
        else:
            ready = planned(document, image, megapixels=megapixels, zoom=sample_zoom)
    except (ValueError, OSError) as error:
        return chips(str(error)), ""
    plan = ready.plan
    tokens = token_count(ready.prompt)
    over = tokens is not None and tokens > MAX_PROMPT_TOKENS
    if image is None:
        asked = "add a sample page to draw"
    else:
        read = dit_tokens(plan, image, ready.sample_lines, ready.prompt)
        asked = f"asks for {gpu_seconds(read, mode, int(steps), guidance)} s of GPU time"
    parts = [
        f"{len(plan.lines)} line{'s' if len(plan.lines) != 1 else ''}",
        f"{plan.width} x {plan.height} px",
        "" if tokens is None else f"{tokens:,} of {MAX_PROMPT_TOKENS:,} prompt tokens",
        asked,
        *ready.notes,
    ]
    return chips(*parts, warn=over or bool(ready.notes) or image is None, title=ASKED_TITLE), ready.prompt


def pick_sample(chosen: gr.SelectData) -> str:
    """The example sample page clicked, as the sample to draw from."""
    return SAMPLES[chosen.index]


def thumbnail(path: str, width: int = 200) -> Image.Image:
    """A sample page small, for the examples: a few kilobytes in place of the scan."""
    with Image.open(path) as image:
        small = image.convert("RGB")
    small.thumbnail((width, 4 * width), Image.Resampling.LANCZOS)
    return small


def readback_rows(plan: CropPlan, verdicts: list[LineVerdict] | None) -> list[list[str]]:
    """A row per line of the page: its text, what the judge read, the read's error rate and its label's tag."""
    if verdicts is None:
        return [[line.text, "", "", "not read"] for line in plan.lines]
    return [
        [line.text, verdict.read, f"{verdict.cer:.2f}", "" if verdict.passes(MAX_LINE_CER) else "unclear"]
        for line, verdict in zip(plan.lines, verdicts, strict=True)
    ]


def draw_page(
    page: str,
    sample: str | None,
    mode: str = FAST if HAS_STUDENT else GUIDED,
    megapixels: float = MEGAPIXELS,
    steps: int = STEPS,
    guidance: float = GUIDANCE,
    seed: int = 0,
    randomize_seed: bool = True,
    sample_zoom: float = 1.0,
    progress: gr.Progress = gr.Progress(track_tqdm=True),  # noqa: B008
) -> tuple[object, ...]:
    """Draw a synthetic archive page from a layout of boxes, in the paper, ink and hand of a sample page.

    Args:
        page: The page as JSON: ``{"aspect": width over height, "lines": line slots down the page, "boxes": [{"id":
            a name, "kind": "text" | "heading" | "note" | "number" | "rule", "rect": [x0, y0, x1, y1] on 0-1000 of the
            page, "text": the box's text, "printed": false}]}``. A box is a block of line slots; its text is written
            into them, a line break starting a new line.
        sample: A scan of a page of handwriting whose paper, ink and hand the page takes.
        mode: ``fast``: the distilled student, 8 steps without guidance; ``guided``: ``steps`` steps at ``guidance``.
        megapixels: The drawn page's size; the model was trained at 2.
        steps: Denoising steps of a guided page.
        guidance: Classifier-free guidance scale of a guided page; 1 draws without guidance, in half the time.
        seed: The noise seed, used unless ``randomize_seed``.
        randomize_seed: Draw from a fresh seed.
        sample_zoom: Show the model the sample's writing this many times larger than the page's own.

    Returns:
        The editor's value with the drawn page, the page, its image, ALTO and PAGE files and its record (the prompt
        and the settings it was drawn with), a status line, the seed, the prompt the model read, the sample as the
        model saw it, and each line's read-back.
    """
    if not sample:
        raise gr.Error("Add a sample page first: a scan of handwriting whose paper, ink and hand the page will take.")
    try:
        document = Document.parse(page)
        image = sample_image(sample)
        ready = planned(document, image, megapixels=megapixels, zoom=sample_zoom)
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error
    tokens = token_count(ready.prompt)
    if tokens is not None and tokens > MAX_PROMPT_TOKENS:
        raise gr.Error(
            f"The page's prompt is {tokens:,} tokens and the text encoder reads {MAX_PROMPT_TOKENS:,}: "
            "take out some lines or boxes."
        )
    if mode not in (FAST, GUIDED) or (mode == FAST and not HAS_STUDENT):
        raise gr.Error(f"No such mode: {mode!r}.")
    plan = ready.plan
    seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    steps = int(steps)
    drawn = draw_on_gpu(plan, image, ready.sample_lines, steps, float(guidance), seed, mode, progress=progress)

    name = f"diffu-page-{seed}"
    folder = Path(tempfile.mkdtemp(prefix="pagedemo-"))
    picture = folder / f"{name}.jpg"
    drawn.image.save(picture, quality=95)
    verdicts = drawn.verdicts
    judged = verdicts if verdicts is not None else [LineVerdict.of(line.text, line.text) for line in plan.lines]
    labels = crop_labels(plan, judged, max_line_cer=MAX_LINE_CER, image_name=picture.name)
    files: dict[LabelFormat, Path] = {}
    for label_format in LabelFormat:
        files[label_format] = folder / f"{name}.{label_format.value}.xml"
        files[label_format].write_text(export(labels, label_format), encoding="utf-8")
    record = DrawRecord(
        model=f"{DIT_REPO}@{DIT_REVISION}",
        mode=mode,
        steps=STUDENT_STEPS if mode == FAST else steps,
        guidance=1.0 if mode == FAST else float(guidance),
        seed=seed,
        megapixels=megapixels,
        width=plan.width,
        height=plan.height,
        sample_zoom=sample_zoom,
        prompt=ready.prompt,
        page=document,
    )
    record_file = folder / f"{name}.json"
    record_file.write_text(record.model_dump_json(indent=1), encoding="utf-8")

    if verdicts is None:
        read = f"lines not read back ({drawn.read_error}): the labels carry no unclear tags"
    else:
        confirmed = sum(verdict.passes(MAX_LINE_CER) for verdict in verdicts)
        read = f"{confirmed} of {len(verdicts)} lines read back as written"
    line = chips(
        f"seed {seed}",
        f"{plan.width} x {plan.height} px",
        f"{STUDENT_STEPS} steps, distilled" if mode == FAST else f"{steps} steps, guidance {guidance:g}",
        f"drawn in {drawn.draw_seconds:.0f} s",
        read,
        *ready.notes,
        warn=verdicts is None or bool(ready.notes),
    )
    shown_sample = sized_sample(image, SampleSize.LEGIBLE, plan=plan, lines=ready.sample_lines)
    return (
        editor_value(document, data_uri(drawn.image, 1500)),
        drawn.image,
        gr.DownloadButton(value=str(picture), visible=True),
        gr.DownloadButton(value=str(files[LabelFormat.ALTO]), visible=True),
        gr.DownloadButton(value=str(files[LabelFormat.PAGE]), visible=True),
        gr.DownloadButton(value=str(record_file), visible=True),
        line,
        seed,
        ready.prompt,
        shown_sample,
        readback_rows(plan, verdicts),
    )


# Gradio writes each rule here twice, as it stands and under the app's own container, but a rule inside @media only
# under the container: the page's side padding, which sits above that container, is set fluid instead
CSS = """
.gradio-container .main.fillable { padding: 14px clamp(10px, 2.6vw, 32px) 24px !important; }
#pd-header .html-container, #pd-editor .html-container, .pd-note .html-container { padding: 0 !important; }
.pd-header { display: flex; align-items: baseline; justify-content: space-between; gap: 6px 18px; flex-wrap: wrap; margin: 0 0 4px; }
.pd-header h1 { font-size: 1.45rem; font-weight: 600; letter-spacing: -0.01em; margin: 0; white-space: nowrap; }
.pd-header p { flex: 1 1 420px; margin: 0; color: var(--body-text-color-subdued); font-size: 0.9rem; line-height: 1.4; }
.pd-links { display: flex; gap: 6px; }
.pd-links a, .pd-links a:visited { text-decoration: none !important; font-size: 0.8rem; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border-color-primary); color: var(--body-text-color); background: var(--block-background-fill); }
.pd-links a:hover { border-color: var(--body-text-color); }
.pd-chips { display: flex; gap: 5px; flex-wrap: wrap; min-height: 24px; }
.pd-chips span { font-size: 0.78rem; padding: 2px 9px; border-radius: 6px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); }
.pd-warn span:last-child { border-color: #b3261e; color: #b3261e; }
#pd-run { font-size: 1rem; min-height: 44px; }
#pd-result:has(.empty), #pd-seen:has(.empty) { height: 120px !important; }
#pd-result img, #pd-seen img { object-fit: contain; max-height: 78vh; }
#pd-examples .grid-wrap { min-height: 0 !important; height: auto; max-height: none; overflow: visible; }
#pd-examples .grid-container { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; grid-template-rows: none !important; grid-auto-rows: auto !important; gap: 6px; height: auto; align-content: start; }
#pd-examples .gallery-item, #pd-examples .thumbnail-item { width: 100% !important; height: auto !important; }
#pd-examples .thumbnail-item { border-radius: 6px; }
#pd-examples .caption-label { font-size: 0.68rem; padding: 1px 5px; }
#pd-random { display: flex; align-items: center; }
#pd-zoom .head label span { white-space: nowrap; }
#pd-mode .wrap { gap: 6px; }
#pd-mode .wrap label { padding: 5px 10px; }
.pd-downloads { gap: 6px; }
.pd-downloads button, .pd-downloads a { min-width: 0 !important; }
.dark .gradio-container { color: var(--body-text-color); }
@media (max-width: 640px) {
  .pd-header h1 { font-size: 1.25rem; }
  .pd-header p { font-size: 0.82rem; }
}
"""

THEME = gr.themes.Soft(
    primary_hue="amber",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
).set(
    button_primary_background_fill="#4a3b22",
    button_primary_background_fill_hover="#66532f",
    button_primary_text_color="#fbf6e9",
    button_primary_border_color="#4a3b22",
    button_primary_background_fill_dark="#7a6337",
    button_primary_background_fill_hover_dark="#8d7443",
    button_primary_text_color_dark="#fbf6e9",
    button_primary_border_color_dark="#7a6337",
    slider_color="#4a3b22",
    slider_color_dark="#a08b62",
    block_title_background_fill="transparent",
    block_title_background_fill_dark="transparent",
    block_title_text_color="*neutral_600",
    block_title_text_color_dark="*neutral_300",
    block_title_padding="0",
    block_title_text_weight="600",
    block_label_background_fill="*neutral_100",
    block_label_background_fill_dark="*neutral_800",
    block_label_text_color="*neutral_600",
    block_label_text_color_dark="*neutral_300",
    checkbox_label_background_fill_selected="#efe6d0",
    checkbox_label_background_fill_selected_dark="#4a3b22",
    checkbox_label_text_color_selected="#2b2317",
    checkbox_label_text_color_selected_dark="#fbf6e9",
    checkbox_label_border_color_selected="#4a3b22",
    checkbox_label_border_color_selected_dark="#a08b62",
    checkbox_background_color_selected="#4a3b22",
    checkbox_background_color_selected_dark="#a08b62",
    checkbox_border_color_selected="#4a3b22",
    checkbox_border_color_selected_dark="#a08b62",
)

HEADER = f"""
<div class="pd-header">
  <h1>diffu · archive pages</h1>
  <p>Lay out a page as boxes or pick a template, write its text, choose a sample page: the model draws the page in
  the sample's paper and scan, with its labels (ALTO and PAGE XML), every line read back.</p>
  <div class="pd-links">
    <a href="https://huggingface.co/{DIT_REPO}" target="_blank">Model ↗</a>
    <a href="https://github.com/Borg93/diffu2" target="_blank">Code ↗</a>
  </div>
</div>
"""

EDITOR_HTML = (HERE / "pagedemo" / "editor.html").read_text(encoding="utf-8")
EDITOR_CSS = (HERE / "pagedemo" / "editor.css").read_text(encoding="utf-8")
EDITOR_JS = (HERE / "pagedemo" / "editor.js").read_text(encoding="utf-8")
SAMPLE_TITLES = {  # an example sample's caption: its years
    "gota-hovratt-1687": "1687",
    "gota-hovratt-1701": "1701",
    "standernas-deputation-1765": "1765",
    "jonkopings-radhusratt-1584-1595": "1590s",
}
SAMPLES = sorted(str(path) for path in (HERE / "samples").glob("*.jpg"))
FIRST = next(iter(TEMPLATES))
TEMPLATES_JSON = json.dumps(
    {name: page.model_dump(mode="json") for name, page in TEMPLATES.items()}, ensure_ascii=False
)


def build() -> gr.Blocks:
    """The demo's page: the sample and the settings in a band on top; under it the editor on the left and, at the
    same size on the right, the button and what it returns."""
    start_mode = FAST if HAS_STUDENT else GUIDED
    start_sample = next((path for path in SAMPLES if "1687" in Path(path).name), None)
    start_note, start_prompt = status(
        editor_value(TEMPLATES[FIRST], None), start_sample, MEGAPIXELS, STEPS, GUIDANCE, start_mode, 1.0
    )
    with gr.Blocks(title="diffu · archive pages") as demo:
        gr.HTML(HEADER, elem_id="pd-header")
        with gr.Row(equal_height=False):
            sample = gr.Image(
                label="Sample",
                type="filepath",
                sources=["upload", "clipboard"],
                height=178,
                value=start_sample,
                scale=0,
                min_width=170,
            )
            examples = gr.Gallery(
                value=[(thumbnail(path), SAMPLE_TITLES.get(Path(path).stem, Path(path).stem)) for path in SAMPLES],
                label="Example samples",
                show_label=False,
                columns=2,
                allow_preview=False,
                object_fit="cover",
                interactive=False,
                buttons=[],
                scale=0,
                min_width=178,
                elem_id="pd-examples",
                visible=bool(SAMPLES),
            )
            with gr.Column(scale=1, min_width=300):
                mode = gr.Radio(
                    label="How it is drawn",
                    choices=[
                        (f"Fast · {STUDENT_STEPS} steps, distilled", FAST),
                        ("Guided · the model itself, slower", GUIDED),
                    ],
                    value=start_mode,
                    visible=HAS_STUDENT,
                    elem_id="pd-mode",
                )
                with gr.Row():
                    megapixels = gr.Dropdown(
                        label="Page size",
                        choices=[("1 MP", 1.0), ("1.5 MP", 1.5), ("2 MP, trained", 2.0)],
                        value=MEGAPIXELS,
                        filterable=False,
                        min_width=170,
                    )
                    steps = gr.Slider(
                        label="Steps",
                        minimum=8,
                        maximum=50,
                        step=1,
                        value=STEPS,
                        buttons=[],
                        min_width=120,
                        elem_id="pd-steps",
                        visible=start_mode == GUIDED,
                    )
                    guidance = gr.Slider(
                        label="Guidance",
                        minimum=1.0,
                        maximum=4.0,
                        step=0.25,
                        value=GUIDANCE,
                        buttons=[],
                        min_width=120,
                        elem_id="pd-guidance",
                        visible=start_mode == GUIDED,
                    )
                    sample_zoom = gr.Slider(
                        label="Sample zoom",
                        minimum=0.5,
                        maximum=3.0,
                        step=0.25,
                        value=1.0,
                        buttons=[],
                        min_width=120,
                        elem_id="pd-zoom",
                    )
                    seed = gr.Number(label="Seed", value=0, precision=0, minimum=0, maximum=MAX_SEED, min_width=110)
                    randomize_seed = gr.Checkbox(
                        label="Random seed", value=True, scale=0, min_width=130, elem_id="pd-random"
                    )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=340):
                editor = gr.HTML(
                    value=editor_value(TEMPLATES[FIRST], None),
                    templates=TEMPLATES_JSON,
                    html_template=EDITOR_HTML,
                    css_template=EDITOR_CSS,
                    js_on_load=EDITOR_JS,
                    elem_id="pd-editor",
                )
            with gr.Column(scale=1, min_width=340):
                run = gr.Button("Draw the page", variant="primary", size="lg", elem_id="pd-run")
                note = gr.HTML(start_note, elem_classes="pd-note")
                with gr.Tabs():
                    with gr.Tab("Page"):
                        outcome = gr.HTML(chips(), elem_classes="pd-note")
                        with gr.Row(elem_classes="pd-downloads"):
                            download_image = gr.DownloadButton("Image", size="sm", visible=False)
                            download_alto = gr.DownloadButton("ALTO XML", size="sm", visible=False)
                            download_page = gr.DownloadButton("PAGE XML", size="sm", visible=False)
                            download_record = gr.DownloadButton("Prompt + settings", size="sm", visible=False)
                        result = gr.Image(
                            label="The drawn page",
                            show_label=False,
                            type="pil",
                            format="jpeg",
                            interactive=False,
                            elem_id="pd-result",
                            buttons=["fullscreen"],
                        )
                    with gr.Tab("Prompt"):
                        prompt = gr.Textbox(
                            label="The prompt the model reads for this page (it follows the editor)",
                            value=start_prompt,
                            lines=14,
                            max_lines=30,
                            interactive=False,
                            buttons=["copy"],
                        )
                    with gr.Tab("Read-back"):
                        readback = gr.Dataframe(
                            headers=["Line", "Read back", "CER", "Label"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Tab("Sample as seen"):
                        seen_sample = gr.Image(
                            label="The sample as the model saw it",
                            show_label=False,
                            type="pil",
                            interactive=False,
                            elem_id="pd-seen",
                        )

        examples.select(fn=pick_sample, outputs=sample, api_visibility="private", show_progress="hidden")
        mode.change(
            fn=lambda chosen: (gr.Slider(visible=chosen == GUIDED), gr.Slider(visible=chosen == GUIDED)),
            inputs=mode,
            outputs=[steps, guidance],
            api_visibility="private",
            show_progress="hidden",
        )
        gr.on(
            triggers=[
                editor.change,
                sample.change,
                megapixels.change,
                steps.release,
                guidance.release,
                mode.change,
                sample_zoom.release,
            ],
            fn=status,
            inputs=[editor, sample, megapixels, steps, guidance, mode, sample_zoom],
            outputs=[note, prompt],
            api_visibility="private",
            show_progress="hidden",
        )
        run.click(
            fn=draw_page,
            inputs=[editor, sample, mode, megapixels, steps, guidance, seed, randomize_seed, sample_zoom],
            outputs=[
                editor,
                result,
                download_image,
                download_alto,
                download_page,
                download_record,
                outcome,
                seed,
                prompt,
                seen_sample,
                readback,
            ],
            api_name="draw_page",
            show_progress_on=result,  # the steps' progress over the page to come, the editor left free
        )
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, mcp_server=True)
