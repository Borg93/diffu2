"""The studio: the page pipeline as a gr.Workflow canvas, a plan node feeding one draw node per drawer.

    diffu-studio --out runs/studio --drawer base=<FLUX.3 dir> --drawer v2=<FLUX.3 dir> --lora v2.v2=<ckpt>/model \\
        --drawer diffu=<diffu repo>

Each drawer (a model directory and its LoRAs) gets a draw node, all fed by one plan, so they draw the same page from
the same noise side by side; drawers on one model share it. The canvas comes with an app view (a form) and a REST
endpoint, ``/<first drawer>_crops``. The graph is rebuilt from the drawers at every start (``<out>/workflow.json``);
edit it on the canvas through the write-access URL ``launch`` prints, and keep that URL private. On the DGX Spark the
studio is a GPU job like any other: start it through ``jobs/spark_guard.sh``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import gradio as gr
import typer

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.cli.options import (
    DEFAULTS,
    Device,
    Guidance,
    Lora,
    TextLayers,
    named_path,
    parse_lora,
    required_model,
    text_layers,
)
from diffu_page.output import console
from diffu_page.pipeline import DrawSettings
from diffu_page.settings import settings
from diffu_studio.graph import PLAN, studio_graph
from diffu_studio.nodes import Drawer, load_draw_nodes, plan_node

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

DEFAULT_STYLE = PageStyle(
    provenance=Provenance(title="the court records of Jämtlands domsaga", years=""),
    look=ScanLook(greyscale=False, paper="cream", ink="dark brown"),
)


def build_studio(drawers: list[Drawer], *, settings: DrawSettings, out: Path) -> gr.Workflow:
    """The studio for ``drawers``: their models loaded, the canvas written to ``<out>/workflow.json``, every node bound."""
    out.mkdir(parents=True, exist_ok=True)
    graph = out / "workflow.json"
    canvas = studio_graph([drawer.name for drawer in drawers], DEFAULT_STYLE)
    graph.write_text(canvas.model_dump_json(indent=1, exclude_none=True), encoding="utf-8")
    nodes = {PLAN: plan_node, **load_draw_nodes(drawers, settings=settings, out=out)}
    return gr.Workflow(graph=str(graph), bind=nodes)


def parse_drawer(value: str) -> Drawer:
    """``name=<model dir>`` as a :class:`Drawer` (its LoRAs come from ``--lora``).

    Raises:
        typer.BadParameter: If ``value`` is not ``name=<dir>``.
    """
    name, model = named_path(value, "name=<model dir>")
    return Drawer(name=name, model=model)


def with_loras(drawers: list[Drawer], loras: list[Lora]) -> list[Drawer]:
    """``drawers`` with each ``<drawer>.<adapter>=<dir>`` LoRA added to its drawer.

    Raises:
        typer.BadParameter: If a LoRA names no drawer or no adapter.
    """
    adapters: dict[str, dict[str, Path]] = {drawer.name: dict(drawer.loras) for drawer in drawers}
    for lora in loras:
        drawer, _, adapter = lora.name.partition(".")
        if drawer not in adapters or not adapter:
            raise typer.BadParameter(
                f"expected <drawer>.<adapter>=<dir> with a drawer of {sorted(adapters)}, got {lora.name!r}"
            )
        adapters[drawer][adapter] = lora.path
    return [drawer.model_copy(update={"loras": adapters[drawer.name]}) for drawer in drawers]


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="where the canvas, the sheets and the zips go")],
    drawer: Annotated[
        list[Drawer] | None,
        typer.Option(parser=parse_drawer, help="name=<model dir> (repeat it; default: flux3=$FLUX3_MODEL)"),
    ] = None,
    lora: Annotated[
        list[Lora] | None, typer.Option(parser=parse_lora, help="<drawer>.<adapter>=<NeMo LoRA model dir>")
    ] = None,
    guidance: Guidance = DEFAULTS.guidance_scale,
    text_layer: TextLayers = None,
    device: Device = DEFAULTS.device,
    port: Annotated[int, typer.Option(help="the port the studio serves on")] = 7860,
) -> None:
    """Serve the studio: one draw node per drawer, all fed by one plan."""
    drawers = with_loras(drawer or [Drawer(name="flux3", model=required_model(settings.flux3_model))], lora or [])
    draw_settings = DrawSettings(guidance_scale=guidance, text_layers=text_layers(text_layer), device=device)
    try:
        studio = build_studio(drawers, settings=draw_settings, out=out)
    except ValueError as error:  # the drawers or settings, refused before a model loads
        raise typer.BadParameter(str(error)) from error
    console.print(f"diffu studio: {', '.join(d.name for d in drawers)} on port {port}, writing to {out}")
    studio.launch(server_port=port, allowed_paths=[str(out)])


if __name__ == "__main__":
    app()
