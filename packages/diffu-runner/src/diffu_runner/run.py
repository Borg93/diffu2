"""A generation run across a Ray cluster: pages planned on CPU, crops drawn by GPU actors, rows written as Parquet.

    diffu-run --jobs run.jsonl --out runs/pages --lora v2=<ckpt>/model --drawers 4 --gpus-per-drawer 1

``run.jsonl`` holds one :class:`diffu_page.jobs.PageJob` per line. Writes ``<out>/crops/*.parquet`` (one
:class:`diffu_runner.stages.DrawnRow` per crop, failed crops with their error) and ``<out>/run_meta.json``.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Annotated

import ray
import typer
from pydantic import BaseModel
from ray.data import Dataset

from diffu_gen.cli.options import Lora, parse_lora
from diffu_gen.generator import GeneratorSettings
from diffu_gen.settings import settings
from diffu_page.jobs import PageJob
from diffu_page.output import console
from diffu_runner.stages import DrawActor, page_rows, plan_rows

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class RunConfig(BaseModel):
    """A run's settings: the model and LoRAs, how crops are drawn, and how many GPU actors draw them."""

    model_config = {"frozen": True}

    model: Path
    loras: dict[str, Path]
    generator: GeneratorSettings
    drawers: int = 1  # GPU actors, each holding one CropGenerator
    gpus_per_drawer: float = 1.0  # several actors share a GPU below 1 (the inference cluster's cards have 96 GB)


class RunMeta(BaseModel):
    """``run_meta.json``: the run's settings, its pages and when it ran (each row carries its own crop's facts)."""

    config: RunConfig
    pages: int
    started: str


def read_pages(jobs: Path) -> list[PageJob]:
    """A run's pages: one :class:`PageJob` per line of ``jobs``."""
    with jobs.open(encoding="utf-8") as lines:
        return [PageJob.model_validate_json(line) for line in lines if line.strip()]


def plan_dataset(pages: list[PageJob]) -> Dataset:
    """The run's crop rows: every page planned into its crops on CPU workers."""
    return ray.data.from_items(page_rows(pages)).flat_map(plan_rows)


def draw_dataset(crops: Dataset, config: RunConfig) -> Dataset:
    """The crop rows drawn by ``config.drawers`` actors (restarted and retried on infrastructure faults)."""
    return crops.map_batches(
        DrawActor,
        fn_constructor_kwargs={"model": config.model, "loras": config.loras, "settings": config.generator},
        batch_size=1,
        concurrency=config.drawers,
        num_gpus=config.gpus_per_drawer,
        max_restarts=3,
        max_task_retries=3,
    )


def run(pages: list[PageJob], config: RunConfig, out: Path) -> None:
    """Plan, draw and write every page's crops to ``out``; connects to the Ray cluster ``ray.init`` finds."""
    if not ray.is_initialized():
        ray.init()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    draw_dataset(plan_dataset(pages), config).write_parquet(str(out / "crops"))
    meta = RunMeta(config=config, pages=len(pages), started=started)
    (out / "run_meta.json").write_text(meta.model_dump_json(indent=1), encoding="utf-8")


@app.command()
def main(
    jobs: Annotated[Path, typer.Option(help="the run's pages: one diffu_page.jobs.PageJob per line")],
    out: Annotated[Path, typer.Option(help="where crops/*.parquet and run_meta.json go")],
    model: Annotated[Path, typer.Option(help="the FLUX.3 model directory")] = settings.flux3_model,
    lora: Annotated[list[Lora] | None, typer.Option(parser=parse_lora, help="name=<NeMo LoRA model dir>")] = None,
    drawers: Annotated[int, typer.Option(help="GPU actors, each holding one generator")] = 1,
    gpus_per_drawer: Annotated[float, typer.Option(help="the share of a GPU each actor takes")] = 1.0,
    steps: int = 50,
    guidance: float = 4.0,
    text_layer: Annotated[
        list[int] | None, typer.Option(help="a text-encoder layer to stack (repeat it; default: the model's own)")
    ] = None,
    device: Annotated[str, typer.Option(help="where the models run")] = "cuda",
) -> None:
    """Generate a run's pages across a Ray cluster."""
    config = RunConfig(
        model=model,
        loras={adapter.name: adapter.path for adapter in lora or []},
        generator=GeneratorSettings(
            steps=steps, guidance_scale=guidance, text_layers=tuple(text_layer) if text_layer else None, device=device
        ),
        drawers=drawers,
        gpus_per_drawer=gpus_per_drawer,
    )
    pages = read_pages(jobs)
    run(pages, config, out)
    console.print(f"{len(pages)} pages -> {out}")


if __name__ == "__main__":
    app()
