"""Draw a MOSS-Image reference training's own pairs back with the model it trained, beside their real pages: the overfit
test of ``jobs/moss/run_ref_full.sh``, where a DiT trained on a handful of pairs should reproduce them.

The pairs are the cache's first ``--limit-pairs`` drawn with their reference, the handful ``diffu-moss-train-ref
--limit-pairs`` trains on (:func:`diffu_moss.train.cache.with_reference`). Each is drawn as it was trained: from its
cached Layout prompt (kept as it is, not compiled again) and its reference image, at its target's size, by the model's
own blocks at its own guidance and sampling shift, from seed ``--seed`` + its place. Writes ``generated/NNN.png`` and
a contact sheet a pair (``NNN_<pair>.jpg``: the reference, the real page, the drawing).

    python -m diffu_page.cli.draw_pairs --model runs/moss/ref_full_overfit --cache data_flux3/cache_moss_ref_v3d \\
        --limit-pairs 8 --out runs/moss_ref_full/overfit_drawn
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import torch
import typer
from PIL import Image

from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.train.cache import read_entries, with_reference
from diffu_moss.train.pairs import read_pairs
from diffu_moss.train.records import CACHE_RECORD, CacheRecord
from diffu_page.cli.options import DEFAULTS, Device, Steps
from diffu_page.output import JPEG_QUALITY, console, load_rgb, side_by_side
from diffu_page.pipeline import DrawSettings, load_pipeline, model_family

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

SHEET_GAP = 8  # px between the panels of a contact sheet


def draw_pairs(model: Path, cache: Path, out: Path, *, count: int, seed: int, settings: DrawSettings) -> list[Path]:
    """The cache's first ``count`` pairs drawn with their reference, drawn back by the MOSS-Image model at ``model``
    into ``out`` (the module's docstring).

    Returns:
        Each pair's contact sheet, in the cache's order.

    Raises:
        ValueError: If ``model`` is not a MOSS-Image model, or the cache is not finished or holds no pair drawn with its
            reference; from drawing, if the model takes no reference image.
    """
    if model_family(model).pipeline is not MossModularPipeline:
        raise ValueError(f"{model} is not a MOSS-Image model: a reference training's pairs are MOSS's Layout prompts")
    entries = with_reference(read_entries(cache), count)
    if not entries:
        raise ValueError(f"{cache} holds no pair drawn with its reference")
    pairs = {pair.id: pair for pair in read_pairs(Path(CacheRecord.read(cache / CACHE_RECORD).pairs))}
    pipe = load_pipeline(model, lambda family: family.blocks(), loras={}, settings=settings)
    (out / "generated").mkdir(parents=True, exist_ok=True)
    sheets = []
    for index, entry in enumerate(entries):
        pair = pairs[entry.pair]
        reference = load_rgb(pair.reference)
        width, height = entry.target_size
        pipe.guider.set_state(step=0, num_inference_steps=settings.steps, timestep=None)  # as diffu_page.pipeline draws
        drawn: Image.Image = pipe(
            prompt=entry.prompt,
            image=reference,
            height=height,
            width=width,
            generator=torch.Generator().manual_seed(seed + index),
            **settings.pipeline_inputs(),
            output="images",
        )[0]
        drawn.save(out / "generated" / f"{index:03d}.png")
        sheet = out / f"{index:03d}_{entry.pair}.jpg"
        side_by_side([reference, load_rgb(pair.target), drawn], height=height, gap=SHEET_GAP).save(
            sheet, quality=JPEG_QUALITY
        )
        sheets.append(sheet)
        console.print(f"{index:3d} {entry.pair} ({entry.variant}) -> {sheet}")
    return sheets


@app.command()
def main(
    model: Annotated[Path, typer.Option(help="a MOSS-Image model directory that takes a reference (diffu-moss-init)")],
    cache: Annotated[Path, typer.Option(help="the diffu-moss-cache-pairs cache the model trained on")],
    limit_pairs: Annotated[
        int, typer.Option(min=1, help="draw the cache's first this many pairs drawn with their reference")
    ],
    out: Annotated[Path, typer.Option(help="where generated/ and the contact sheets go")],
    steps: Steps = DEFAULTS.steps,
    seed: Annotated[int, typer.Option(help="pair k is drawn from seed + k")] = 0,
    device: Device = DEFAULTS.device,
) -> None:
    """Draw a reference training's own pairs back beside their real pages: does the model reproduce them?"""
    try:
        sheets = draw_pairs(
            model, cache, out, count=limit_pairs, seed=seed, settings=DrawSettings(steps=steps, device=device)
        )
    except ValueError as error:  # not a MOSS model, or a cache without such pairs: the arguments' fault
        raise typer.BadParameter(str(error)) from error
    console.print(f"{len(sheets)} pairs drawn beside their real pages -> {out}")


if __name__ == "__main__":
    app()
