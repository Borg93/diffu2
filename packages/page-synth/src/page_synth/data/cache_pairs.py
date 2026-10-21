"""Cache the v3 reference pairs (``page_synth.data.style_pairs``'s manifest) with NeMo AutoModel's FLUX.3 image-edit encoder.

NeMo's ``image-edit`` preprocessing CLI reads a Hugging Face dataset and materializes it into a manifest; ours is
already one, so this hands it straight to the encoder that CLI calls (``Flux3ImageEditCacheEncoder``). Targets keep
their exact bucket size (none is over ``LARGEST_BUCKET_PIXELS``); the references were shrunk by ``page_synth.data.style_pairs``.
Runs in the NeMo AutoModel environment, on the GPU:

    cd ~/Desktop/repos/Automodel && python -m page_synth.data.cache_pairs \
        --manifest ~/Desktop/repos/diffu2/data_flux3/style_pairs_v3/train.jsonl --out ~/Desktop/repos/diffu2/data_flux3/cache_style_v3
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from nemo_automodel.components.models.flux3.preprocessing import (
    MAX_SEQUENCE_LENGTH,
    Flux3ImageEditCacheEncoder,
)

from page_synth.cli.output import console
from page_synth.crops.grid import LARGEST_BUCKET_PIXELS
from page_synth.settings import settings

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


def first_rows(manifest: Path, limit: int) -> Path:
    """A manifest of the first ``limit`` rows, beside the full one so its media paths still resolve."""
    subset = manifest.with_name(f"{manifest.stem}_first{limit}.jsonl")
    with manifest.open(encoding="utf-8") as rows:
        subset.write_text("".join(islice(rows, limit)), encoding="utf-8")
    return subset


@app.command()
def main(
    manifest: Annotated[Path, typer.Option(help="page_synth.data.style_pairs' train.jsonl")],
    out: Annotated[Path, typer.Option(help="the cache directory")],
    model: Annotated[Path, typer.Option(help="the FLUX.3 model directory")] = settings.flux3_model,
    limit: Annotated[int | None, typer.Option(help="cache only the first N pairs (a smoke test)")] = None,
) -> None:
    """Cache the v3 reference pairs with NeMo AutoModel's FLUX.3 image-edit encoder."""
    rows = first_rows(manifest, limit) if limit else manifest
    encoder = Flux3ImageEditCacheEncoder(model_name=str(model), max_sequence_length=MAX_SEQUENCE_LENGTH)
    metadata = encoder.encode_manifest(
        manifest_path=rows,
        output_dir=out,
        max_pixels=LARGEST_BUCKET_PIXELS,
        resolution_preset=None,
        num_gpus=1,
        verify=False,
    )
    console.print(f"cached {rows} -> {metadata}")


if __name__ == "__main__":
    app()
