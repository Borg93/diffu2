"""Cache a split of crops for training diffu (``diffu_train.cache``), with NeMo's own bucket ids.

Runs in NeMo AutoModel's environment, on the GPU, through jobs/spark_guard.sh:

    python -m diffu_train.cache_diffu --crops data_flux3/layout_crops_v2/train --model runs/diffu/init_s \\
        --out data_flux3/cache_diffu_v2

Writes ``<out>/failed.json`` with the crops that could not be cached, and exits 1 if there are any. When nothing was
cached it writes no ``metadata.json``, the file a finished cache is known by.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from nemo_automodel.components.datasets.diffusion.multi_tier_bucketing import MultiTierBucketCalculator

from diffu_page.captions.records import read_records
from diffu_page.output import console, err_console
from diffu_train.cache import DiffuEncoders, cache_crops, write_empty_prompt, write_metadata

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def main(
    crops: Annotated[Path, typer.Option(help="a split of diffu_page.data.build_crops: metadata.jsonl and the crops")],
    model: Annotated[Path, typer.Option(help="the diffu model repo whose encoders to cache with (diffu-init's)")],
    out: Annotated[Path, typer.Option(help="the cache directory NeMo's dataloader reads")],
    limit: Annotated[
        int | None, typer.Option(min=1, help="cache only the first N crops (an overfit set or a smoke run)")
    ] = None,
    device: Annotated[str, typer.Option(help="where the encoders run")] = "cuda",
) -> None:
    """Cache a split's crops for diffu: latents and prompt embeddings from diffu's own encoders."""
    records = read_records(crops / "metadata.jsonl")[:limit]
    calculator = MultiTierBucketCalculator(quantization=64)
    out.mkdir(parents=True, exist_ok=True)
    encoders = DiffuEncoders.load(model, device)
    write_empty_prompt(encoders, out)  # first: seconds, where the crops take hours
    run = cache_crops(
        records,
        crops,
        encoders=encoders,
        out=out,
        bucket_id=lambda width, height: calculator.get_bucket_for_image(width, height)["id"],
    )
    (out / "failed.json").write_text(json.dumps(run.failed, indent=1), encoding="utf-8")
    if run.failed:
        err_console.print(f"{len(run.failed)} of {len(records)} crops failed: see {out / 'failed.json'}")
    if not run.entries:  # no metadata.json: it marks a finished cache
        raise typer.Exit(code=1)
    write_metadata(run.entries, out, model=model)
    console.print(f"cached {len(run.entries)} of {len(records)} crops -> {out}")
    if run.failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
