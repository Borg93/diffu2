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
from diffu_page.page.labels import LabelFormat, export  # noqa: E402
from diffu_page.quality.judge import MAX_LINE_CER, LineVerdict  # noqa: E402
from pagedemo.document import Document  # noqa: E402
from pagedemo.drawing import DIT_REPO, Drawn, Models, draw, load_models  # noqa: E402
from pagedemo.planning import layout_prompt_of, page_plan, planned  # noqa: E402
from pagedemo.templates import TEMPLATES  # noqa: E402

HERE = Path(__file__).parent
STUB = os.environ.get("PAGEDEMO_STUB") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = 2**31 - 1
STEPS, GUIDANCE, MEGAPIXELS = 28, 2.0, 2.0  # a page's defaults: the guidance the model card suggests, at its size
# The GPU seconds a page is asked for: the model's start on a GPU and its text encoder, VAE and judge, then each
# pass of the DiT by the megapixels it draws (two passes a step under guidance)
SETUP_SECONDS, SECONDS_PER_PASS_MEGAPIXEL, LONGEST_SECONDS = 30, 1.1, 300

MODELS: Models | None = None if STUB else load_models(DEVICE, student=HERE / "student")
FAST, GUIDED = "fast", "guided"  # a page's mode: the distilled student's few unguided steps, or the DiT's guided ones
STUDENT_STEPS = 8 if MODELS is None or MODELS.student is None else MODELS.student.student_steps
HAS_STUDENT = STUB or (MODELS is not None and MODELS.student is not None)


def passes(mode: str, steps: int, guidance: float) -> int:
    """The DiT passes a page takes: the student's steps, or the steps asked, twice under guidance."""
    if mode == FAST:
        return STUDENT_STEPS
    return steps * (1 if math.isclose(guidance, 1.0) else 2)


def gpu_seconds(width: int, height: int, dit_passes: int) -> int:
    """The GPU seconds asked for a page of ``width`` x ``height`` px drawn in ``dit_passes`` passes of the DiT."""
    seconds = SETUP_SECONDS + dit_passes * width * height / 1_000_000 * SECONDS_PER_PASS_MEGAPIXEL
    return min(LONGEST_SECONDS, math.ceil(seconds))


def _duration(
    plan: CropPlan, sample: object, lines: object, steps: int, guidance: float, seed: int, mode: str, *_: object
) -> int:
    return gpu_seconds(plan.width, plan.height, passes(mode, steps, guidance))


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


def chips(*parts: str, warn: bool = False) -> str:
    """A status line of small labels."""
    kind = "pd-chips pd-warn" if warn else "pd-chips"
    return f"<div class='{kind}'>" + "".join(f"<span>{html.escape(part)}</span>" for part in parts if part) + "</div>"


def token_count(prompt: str) -> int | None:
    """The tokens the text encoder reads for ``prompt``; None in the stub, which has no tokenizer."""
    return None if MODELS is None else prompt_tokens(MODELS.pipe.tokenizer, prompt)


def status(value: str, megapixels: float, steps: int, guidance: float, mode: str) -> str:
    """What the page in the editor comes to before it is drawn: its lines, its prompt's length and its GPU time."""
    try:
        plan, notes = page_plan(Document.parse(value), megapixels=megapixels)
    except ValueError as error:
        return chips(str(error))
    tokens = token_count(layout_prompt_of(plan))
    over = tokens is not None and tokens > MAX_PROMPT_TOKENS
    parts = [
        f"{len(plan.lines)} line{'s' if len(plan.lines) != 1 else ''}",
        f"{plan.width} x {plan.height} px",
        "" if tokens is None else f"{tokens:,} of {MAX_PROMPT_TOKENS:,} prompt tokens",
        f"about {gpu_seconds(plan.width, plan.height, passes(mode, int(steps), guidance))} s of GPU time",
        *notes,
    ]
    return chips(*parts, warn=over or bool(notes))


def load_template(name: str) -> str:
    """A template as the editor's value, with no drawn page under it."""
    return editor_value(TEMPLATES[name], None)


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
        The editor's value with the drawn page, the page, its image, ALTO and PAGE files, a status line, the seed, the
        prompt the model read, the sample as the model saw it, and each line's read-back.
    """
    if not sample:
        raise gr.Error("Add a sample page first: a scan of handwriting whose paper, ink and hand the page will take.")
    try:
        document = Document.parse(page)
        image = load_rgb(Path(sample))
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
        line,
        seed,
        ready.prompt,
        shown_sample,
        readback_rows(plan, verdicts),
    )


CSS = """
#pd-app { max-width: 1320px; margin: 0 auto; padding: 8px 4px 64px; }
.pd-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin: 4px 0 14px; }
.pd-header h1 { font-size: 1.9rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 6px; }
.pd-header p { margin: 0; max-width: 780px; color: var(--body-text-color-subdued); font-size: 0.97rem; line-height: 1.5; }
.pd-links { display: flex; gap: 8px; flex-wrap: wrap; }
.pd-links a, .pd-links a:visited { text-decoration: none !important; font-size: 0.85rem; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-color-primary); color: var(--body-text-color); background: var(--block-background-fill); }
.pd-links a:hover { border-color: var(--body-text-color); }
.pd-templates { flex-wrap: wrap; gap: 8px; margin: 0 0 6px; align-items: center; }
.pd-templates button { flex: 0 1 auto !important; min-width: 0 !important; }
.pd-label { font-size: 0.82rem; color: var(--body-text-color-subdued); margin: 0; }
.pd-chips { display: flex; gap: 6px; flex-wrap: wrap; min-height: 26px; }
.pd-chips span { font-size: 0.8rem; padding: 3px 10px; border-radius: 6px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); }
.pd-warn span:last-child { border-color: #b3261e; color: #b3261e; }
#pd-run { font-size: 1.05rem; min-height: 48px; }
#pd-result img { object-fit: contain; }
.dark .gradio-container { color: var(--body-text-color); }
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
    slider_color="#4a3b22",
)

HEADER = f"""
<div class="pd-header">
  <div>
    <h1>diffu · archive pages</h1>
    <p>Lay out a page as boxes, or start from a template, write its text, and give a sample page: the model draws the
    page in the sample's paper, ink and hand, and returns it with its labels (ALTO and PAGE XML), every line read back.
    A fine-tune of MOSS-Image for Swedish archival handwriting; it writes the text you give it, line for line.</p>
  </div>
  <div class="pd-links">
    <a href="https://huggingface.co/{DIT_REPO}" target="_blank">Model ↗</a>
    <a href="https://github.com/Borg93/diffu2" target="_blank">Code ↗</a>
  </div>
</div>
"""

EDITOR_HTML = (HERE / "pagedemo" / "editor.html").read_text(encoding="utf-8")
EDITOR_CSS = (HERE / "pagedemo" / "editor.css").read_text(encoding="utf-8")
EDITOR_JS = (HERE / "pagedemo" / "editor.js").read_text(encoding="utf-8")
SAMPLES = sorted(str(path) for path in (HERE / "samples").glob("*.jpg"))
FIRST_SAMPLE = next((path for path in SAMPLES if "1687" in Path(path).name), SAMPLES[0] if SAMPLES else None)
FIRST = next(iter(TEMPLATES))


def build() -> gr.Blocks:
    """The demo's page."""
    with gr.Blocks(title="diffu · archive pages", elem_id="pd-app") as demo:
        gr.HTML(HEADER)
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                gr.HTML("<p class='pd-label'>Start from a template, or draw the page's boxes yourself</p>")
                with gr.Row(elem_classes="pd-templates"):
                    template_buttons = [gr.Button(name, size="sm", variant="secondary") for name in TEMPLATES]
                editor = gr.HTML(
                    value=load_template(FIRST),
                    html_template=EDITOR_HTML,
                    css_template=EDITOR_CSS,
                    js_on_load=EDITOR_JS,
                    elem_id="pd-editor",
                )
            with gr.Column(scale=5):
                sample = gr.Image(
                    label="Sample page: its paper, ink and hand",
                    type="filepath",
                    sources=["upload", "clipboard"],
                    height=300,
                    value=FIRST_SAMPLE,
                )
                if SAMPLES:
                    gr.Examples(
                        examples=[[path] for path in [FIRST_SAMPLE, *(p for p in SAMPLES if p != FIRST_SAMPLE)]],
                        inputs=[sample],
                        label="Samples (Swedish National Archives, public HTR datasets)",
                        cache_examples=False,
                    )
                mode = gr.Radio(
                    label="How it is drawn",
                    choices=[
                        (f"Fast: {STUDENT_STEPS} steps, a distilled student of the model", FAST),
                        ("Guided: the model's own steps, several times the GPU time", GUIDED),
                    ],
                    value=FAST if HAS_STUDENT else GUIDED,
                    visible=HAS_STUDENT,
                )
                run = gr.Button("Draw the page", variant="primary", size="lg", elem_id="pd-run")
                note = gr.HTML(status(load_template(FIRST), MEGAPIXELS, STEPS, GUIDANCE, mode.value))
                result = gr.Image(
                    label="The drawn page",
                    type="pil",
                    format="jpeg",
                    interactive=False,
                    elem_id="pd-result",
                    buttons=["fullscreen"],
                    height=460,
                )
                outcome = gr.HTML(chips())
                with gr.Row():
                    download_image = gr.DownloadButton("Image", size="sm", visible=False)
                    download_alto = gr.DownloadButton("ALTO XML", size="sm", visible=False)
                    download_page = gr.DownloadButton("PAGE XML", size="sm", visible=False)
                with gr.Accordion("Settings", open=False):
                    megapixels = gr.Radio(
                        label="Page size, megapixels",
                        info="The model was trained at 2; smaller pages draw faster and less surely.",
                        choices=[1.0, 1.5, 2.0],
                        value=MEGAPIXELS,
                    )
                    steps = gr.Slider(label="Steps of a guided page", minimum=8, maximum=50, step=1, value=STEPS)
                    guidance = gr.Slider(
                        label="Guidance of a guided page",
                        info="Higher holds closer to the text but darkens the ink; 1 draws in half the time.",
                        minimum=1.0,
                        maximum=4.0,
                        step=0.25,
                        value=GUIDANCE,
                    )
                    sample_zoom = gr.Slider(
                        label="Sample zoom",
                        info="1 shows the model the sample at the page's own line height; higher shows its pen larger.",
                        minimum=0.5,
                        maximum=3.0,
                        step=0.25,
                        value=1.0,
                    )
                    with gr.Row():
                        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, scale=3)
                        randomize_seed = gr.Checkbox(label="Randomize", value=True, scale=1, min_width=100)
                with gr.Accordion("What the model was given, and what was read back", open=False):
                    seen_sample = gr.Image(
                        label="The sample as the model saw it", type="pil", interactive=False, height=260
                    )
                    prompt = gr.Textbox(label="The prompt", lines=4, max_lines=12, interactive=False)
                    readback = gr.Dataframe(
                        headers=["Line", "Read back", "CER", "Label"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                    )

        for name, button in zip(TEMPLATES, template_buttons, strict=True):
            button.click(fn=lambda name=name: load_template(name), outputs=editor, api_visibility="private")
        gr.on(
            triggers=[editor.change, megapixels.change, steps.release, guidance.release, mode.change],
            fn=status,
            inputs=[editor, megapixels, steps, guidance, mode],
            outputs=note,
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
                outcome,
                seed,
                prompt,
                seen_sample,
                readback,
            ],
            api_name="draw_page",
        )
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, mcp_server=True)
