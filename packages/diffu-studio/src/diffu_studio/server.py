"""The ``gradio.Server`` app: the Blocks UI mounted at ``/ui`` + streaming ``@app.api`` SSE endpoints.

One process serves both the human control panel (``build_blocks``) and a programmatic API over the exact
same core, so a future custom frontend (``@gradio/client`` → ``app.predict("/stream_page", …)``) needs no
new logic. ``@app.get("/")`` redirects to the UI for now and is the slot that custom frontend will take.
GPU entry points are wrapped in :func:`gpu_task` so a ZeroGPU Space allocates a GPU for each stream.
"""

from collections.abc import Iterator

import gradio as gr
from PIL import Image

from diffu_studio.checkpoints import CheckpointMismatch
from diffu_studio.configs import SampleSettings, build_gen_config, default_configs, list_templates
from diffu_studio.core import list_checkpoints
from diffu_studio.device import gpu_task, torch_dtype
from diffu_studio.engine import StudioEngine
from diffu_studio.line import stream_line, style_from_pil
from diffu_studio.page import PageRequest, stream_page
from diffu_studio.ui import build_blocks


def build_server(
    engine: StudioEngine,
    *,
    gallery_paths: list[str] | None,
    default_ckpt: str | None,
) -> gr.Server:
    """Build the ``gradio.Server``: SSE endpoints over the core + the Blocks UI mounted at ``/ui``."""
    app = gr.Server(title="Diffu Studio", description="Single-line + whole-page handwriting synthesis.")

    def _style(style_path: str):
        return style_from_pil(
            Image.open(style_path).convert("RGB"), engine.device.torch_device, torch_dtype(engine.device)
        )

    @app.api(name="list_checkpoints")
    def list_checkpoints_api() -> list[dict]:
        return [c.model_dump() for c in list_checkpoints(engine.ckpt_root)]

    @app.api(name="list_templates")
    def list_templates_api() -> list[str]:
        return list_templates()

    @app.api(name="default_configs")
    def default_configs_api() -> dict:
        return default_configs()

    @app.api(name="stream_line")
    @gpu_task
    def stream_line_api(
        text: str,
        style_path: str,
        checkpoint: str,
        steps: int = 24,
        cfg_scale: float = 5.0,
        auto_width: bool = True,
    ) -> Iterator[tuple[str, Image.Image | None]]:
        settings = SampleSettings(steps=steps, cfg_scale=cfg_scale, auto_width=auto_width)
        try:
            loaded = engine.model(checkpoint)
        except CheckpointMismatch as exc:
            yield f"⚠ {exc}", None
            return
        for step in stream_line(loaded, _style(style_path), text, settings, recognizer=engine.recognizer):
            yield (step.read or step.status), step.image

    @app.api(name="stream_page")
    @gpu_task
    def stream_page_api(
        text: str,
        style_path: str,
        checkpoint: str,
        template: str = "single_column",
        steps: int = 24,
        cfg_scale: float = 5.0,
        seed: int = 0,
    ) -> Iterator[tuple[str, Image.Image | None, dict | None]]:
        gen_cfg = build_gen_config(template=template, steps=steps, cfg_scale=cfg_scale)
        request = PageRequest(
            paragraphs=[p for p in text.split("\n") if p.strip()] or [text],
            gen=gen_cfg,
            settings=SampleSettings(steps=steps, cfg_scale=cfg_scale),
            seed=seed,
            animate=engine.device.interactive,
        )
        try:
            loaded = engine.model(checkpoint)
        except CheckpointMismatch as exc:
            yield f"⚠ {exc}", None, None
            return
        for step in stream_page(loaded, _style(style_path), request):
            yield step.status, step.page, step.labels

    blocks = build_blocks(engine, gallery_paths=gallery_paths, default_ckpt=default_ckpt)
    gr.mount_gradio_app(app, blocks, path="/ui")

    @app.get("/")
    def index():  # reserved for the future custom frontend; redirect to the mounted UI for now
        from starlette.responses import RedirectResponse

        return RedirectResponse("/ui")

    return app
