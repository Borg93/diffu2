"""The studio: a template page, a text and a style in; every crop of the page drawn, with its labels, out.

    diffu-studio --model <model dir> --lora v2=<ckpt>/model --out runs/studio

The page pipeline (:func:`diffu_page.pipeline.page_blocks`) is loaded once, then each request is drawn with it. A
request's crops and their labels are also written to ``<out>/<template>_<seed>.zip`` for download.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from statistics import fmean
from typing import Annotated, Any

import gradio as gr
import typer
from diffusers.modular_pipelines import ModularPipeline
from gradio.blocks import Block
from PIL import Image
from pydantic import BaseModel, ConfigDict

from diffu_page.blocks import DrawnCrop
from diffu_page.captions.look import ScanLook
from diffu_page.captions.provenance import Provenance
from diffu_page.cli.options import Lora, parse_lora
from diffu_page.layout import load_xml_layout, running_text
from diffu_page.output import console
from diffu_page.pipeline import DrawSettings, load_pipeline, page_blocks
from diffu_page.plan import PageStyle
from diffu_page.settings import settings

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

type Gallery = list[tuple[Image.Image, str]]


class PageRequest(BaseModel):
    """One request from the form: a template page, the text to write into it (blank: its own), the style, the noise
    seed and the denoising steps."""

    template: Path
    text: str
    title: str
    years: str
    paper: str
    ink: str
    greyscale: bool
    seed: int
    steps: int

    def style(self) -> PageStyle:
        """The style the request names."""
        return PageStyle(
            provenance=Provenance(title=self.title, years=self.years),
            look=ScanLook(greyscale=self.greyscale, paper=self.paper, ink=self.ink),
        )


class DrawnPage(BaseModel):
    """A drawn page as the form shows it: every crop with its caption, and the zip of the crops with their labels."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gallery: Gallery
    archive: Path


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _caption(crop: DrawnCrop) -> str:
    return f"{crop.name}: read-back CER {fmean(crop.line_cers):.3f}, {crop.unclear_lines} unclear lines"


def draw_page(pipe: ModularPipeline, request: PageRequest, *, settings: DrawSettings, out: Path) -> DrawnPage:
    """Every crop of the request's page drawn by the page pipeline, and zipped with its labels into ``out``."""
    text = request.text.strip() or running_text(load_xml_layout(request.template))
    crops: list[DrawnCrop] = pipe(
        template=request.template,
        text=text,
        style=request.style(),
        page_name=request.template.stem,
        seed=request.seed,
        **settings.model_copy(update={"steps": request.steps}).pipeline_inputs(),
        output="crops",
    )
    archive = out / f"{request.template.stem}_{request.seed}.zip"
    with zipfile.ZipFile(archive, "w") as files:
        for crop in crops:
            files.writestr(f"{crop.name}.jpg", _jpeg(crop.image))
            files.writestr(f"{crop.name}.xml", crop.labels_xml)
    return DrawnPage(gallery=[(crop.image, _caption(crop)) for crop in crops], archive=archive)


def build_studio(pipe: ModularPipeline, *, settings: DrawSettings, out: Path) -> gr.Blocks:
    """The studio's form around ``pipe``, a loaded page pipeline; its ``/draw`` endpoint takes the form's fields by
    name (:class:`PageRequest`)."""
    with gr.Blocks(title="diffu studio", analytics_enabled=False) as studio:
        gr.Markdown("# diffu studio\nA real page's layout, any text, a chosen archive's look: crops with labels.")
        with gr.Row():
            with gr.Column():
                fields: dict[str, Block] = {
                    "template": gr.File(label="Template (PAGE or ALTO XML)", file_types=[".xml"], type="filepath"),
                    "text": gr.Textbox(label="Text (blank: the template's own)", lines=6),
                    "title": gr.Textbox(label="Archive", value="the court records of Jämtlands domsaga"),
                    "years": gr.Textbox(label="Years", value=""),
                    "paper": gr.Textbox(label="Paper", value="cream"),
                    "ink": gr.Textbox(label="Ink", value="dark brown"),
                    "greyscale": gr.Checkbox(label="Greyscale scan", value=False),
                    "seed": gr.Number(label="Seed", value=0, precision=0),
                    "steps": gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=settings.steps),
                }
                draw = gr.Button("Draw", variant="primary")
            with gr.Column():
                gallery = gr.Gallery(label="Crops")
                archive = gr.File(label="Crops and labels (zip)")

        def on_draw(**values: Any) -> tuple[Gallery, str]:
            page = draw_page(pipe, PageRequest.model_validate(values), settings=settings, out=out)
            return page.gallery, str(page.archive)

        draw.click(on_draw, inputs_kwargs=fields, outputs=[gallery, archive], api_name="draw")
    return studio


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="where each request's zip of crops and labels goes")],
    model: Annotated[Path, typer.Option(help="the model directory (FLUX.3 Klein or diffu)")] = settings.flux3_model,
    lora: Annotated[list[Lora] | None, typer.Option(parser=parse_lora, help="name=<NeMo LoRA model dir>")] = None,
    guidance: float = 4.0,
    text_layer: Annotated[
        list[int] | None, typer.Option(help="a text-encoder layer to stack (repeat it; default: the model's own)")
    ] = None,
    device: Annotated[str, typer.Option(help="where the models run")] = "cuda",
    port: Annotated[int, typer.Option(help="the port the studio serves on")] = 7860,
) -> None:
    """Serve the studio on a model and its LoRAs."""
    draw_settings = DrawSettings(
        guidance_scale=guidance, text_layers=tuple(text_layer) if text_layer else None, device=device
    )
    loras = {adapter.name: adapter.path for adapter in lora or []}
    pipe = load_pipeline(model, page_blocks, loras=loras, settings=draw_settings)
    out.mkdir(parents=True, exist_ok=True)
    console.print(f"diffu studio on port {port}, writing to {out}")
    build_studio(pipe, settings=draw_settings, out=out).launch(server_port=port)


if __name__ == "__main__":
    app()
