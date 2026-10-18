"""Single-line mode: render one text line in the reference hand and read it back with TrOCR.

Model selection is gone (one fixed checkpoint), so this collapses to a single image + readback — no A/B compare.
"""

import gradio as gr

from diffu_studio.controls import LineControls
from diffu_studio.device import gpu_task
from diffu_studio.engine import StudioEngine
from diffu_studio.handlers import stream_line_view


def build_line_view(
    *, engine: StudioEngine, default_ckpt: str, ref_state: gr.Image, text: gr.Textbox
) -> None:
    """Build the single-line controls + preview and wire Generate. ``ref_state`` and ``text`` are shared inputs."""
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Row():
                steps = gr.Slider(
                    4,
                    50,
                    value=24,
                    step=1,
                    label="Steps",
                    info="Denoise iterations — higher is smoother but slower.",
                )
                cfg = gr.Slider(
                    0.0,
                    12.0,
                    value=5.0,
                    step=0.5,
                    label="CFG",
                    info="Text-following strength; ~5 is the legible sweet spot, higher can distort.",
                )
            with gr.Row():
                auto = gr.Checkbox(
                    value=True,
                    label="Auto-width",
                    info="Size the canvas to the text, then ink-crop the blank tail.",
                )
                width = gr.Slider(
                    128,
                    1024,
                    value=448,
                    step=16,
                    label="Manual width (px)",
                    info="Fixed canvas width — only used when Auto-width is off.",
                )
            with gr.Row():
                lora = gr.Textbox(
                    value="",
                    label="LoRA adapter",
                    placeholder="runs/iam_lora/final",
                    info="Optional Stage-3 adapter (dir or Hub id) applied on top of the checkpoint.",
                )
                lora_scale = gr.Slider(
                    0.0,
                    1.5,
                    value=1.0,
                    step=0.05,
                    label="LoRA scale",
                    info="Adapter strength — 0 = base checkpoint, 1 = full.",
                )
            go = gr.Button("✨ Generate", variant="primary", size="lg")
            status = gr.Markdown()
        with gr.Column(scale=1):
            out = gr.Image(label="Generated line", height=150)
            read = gr.Markdown()

    controls = LineControls({"text": text, "steps": steps, "cfg": cfg, "auto": auto, "width": width})

    # ZeroGPU fork-safety: pass plain VALUES in a fixed order (see page_view) — component-keyed dicts don't
    # survive the @spaces.GPU pickle boundary. lora/lora_scale ride alongside as model-load knobs (like ckpt).
    line_inputs: list = [ref_state, lora, lora_scale, *controls.components()]

    def on_line(*vals):
        data = dict(zip(line_inputs, vals, strict=True))
        yield from stream_line_view(
            engine,
            controls.read(data),
            ref=data[ref_state],
            ckpt=default_ckpt,
            lora=(data[lora] or "").strip() or None,
            lora_scale=float(data[lora_scale]),
        )

    go.click(gpu_task(on_line), inputs=line_inputs, outputs=[out, read, status])
