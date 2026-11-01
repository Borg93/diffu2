"""The studio: the page pipeline as a gr.Workflow canvas, a plan node feeding one draw node per MOSS-Image drawer.

    diffu-studio --out runs/studio --drawer base=runs/moss/base --drawer hand=runs/moss/hand_1715

Each drawer (a MOSS-Image model directory, ``diffu-moss-init``'s) gets a draw node, all fed by one plan, so they draw
the same page from the same noise side by side. The canvas comes with an app view (a form) and a REST endpoint,
``/<first drawer>_crops``. The graph is rebuilt from the drawers at every start (``<out>/workflow.json``); edit it on
the canvas through the write-access URL ``launch`` prints, and keep that URL private. On the DGX Spark the studio is a
GPU job like any other: start it through ``jobs/spark_guard.sh``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import gradio as gr
import typer

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.cli.options import DEFAULTS, Device, Guidance, named_path
from diffu_page.output import console
from diffu_page.pipeline import DrawSettings
from diffu_studio.graph import PLAN, studio_graph
from diffu_studio.nodes import Drawer, check_drawers, load_draw_nodes, plan_node

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
    """``name=<model dir>`` as a :class:`Drawer`.

    Raises:
        typer.BadParameter: If ``value`` is not ``name=<dir>``.
    """
    name, model = named_path(value, "name=<MOSS-Image model dir>")
    return Drawer(name=name, model=model)


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="where the canvas, the sheets and the zips go")],
    drawer: Annotated[list[Drawer], typer.Option(parser=parse_drawer, help="name=<MOSS-Image model dir> (repeat it)")],
    guidance: Guidance = DEFAULTS.guidance_scale,
    device: Device = DEFAULTS.device,
    port: Annotated[int, typer.Option(help="the port the studio serves on")] = 7860,
) -> None:
    """Serve the studio: one draw node per drawer, all fed by one plan."""
    try:
        check_drawers(drawer)
    except ValueError as error:  # the drawers, refused before a model loads
        raise typer.BadParameter(str(error), param_hint="--drawer") from error
    studio = build_studio(drawer, settings=DrawSettings(guidance_scale=guidance, device=device), out=out)
    console.print(f"diffu studio: {', '.join(d.name for d in drawer)} on port {port}, writing to {out}")
    studio.launch(server_port=port, allowed_paths=[str(out)])


if __name__ == "__main__":
    app()
