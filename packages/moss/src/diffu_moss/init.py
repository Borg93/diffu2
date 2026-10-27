"""A MOSS-Image model repo: what the page pipeline draws with (``diffu-generate --model <out>``) and
``MossModularPipeline.from_pretrained`` loads.

    diffu-moss-init --base <MOSS-Image-Base snapshot> --out runs/moss/base
    diffu-moss-init --base <MOSS-Image-Base snapshot> --transformer <export> --out runs/moss/hand_1715

Writes ``<out>/modular_model_index.json``: the pipeline and its blocks (``MossAutoBlocks``), the DiT blocks the
unconditional pass of guidance skips, and where each component loads from: the text encoder (Qwen3.5-9B), its
tokenizer and the VAE from the Base snapshot's folders, the DiT from ``--transformer`` (a diffusers-format folder, its
``config.json`` and safetensors: a LoRA merged into Base, or a full fine-tune, as KohakuMossTrainer exports them) or
the snapshot's own. Nothing is copied: the index names the folders where they are. The scheduler is MOSS-Image's
sampler at its demo's shift, 5 (``--sampling-shift`` changes it per run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import diffusers
import typer
from diffusers import AutoencoderKLFlux2
from rich.console import Console
from transformers import AutoTokenizer, Qwen3_5Model

from diffu.repo import INDEX, ModelIndex, index_entry
from diffu_moss.modular_pipelines.moss import MossAutoBlocks, MossModularPipeline
from diffu_moss.modular_pipelines.moss.denoise import DEMO_UNCOND_SKIP_LAYERS
from diffu_moss.transformer import MossTransformer

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

BASE_FOLDERS = {"text_encoder": "config.json", "tokenizer": "tokenizer_config.json", "vae": "config.json"}


def _require(path: Path, what: str) -> None:
    if not path.is_file():
        raise ValueError(f"{path.parent} is not {what}: it has no {path.name}")


def moss_index(base: Path, transformer: Path, *, uncond_skip_layers: tuple[int, ...]) -> ModelIndex:
    """The ``modular_model_index.json`` of a MOSS-Image repo: its pipeline and block classes, the blocks its
    unconditional pass skips, the text encoder, tokenizer and VAE in the Base snapshot ``base``'s folders and the DiT
    in the folder ``transformer``."""
    snapshot = str(base.resolve())
    return ModelIndex.model_validate(
        {
            "_blocks_class_name": MossAutoBlocks.__name__,
            "_class_name": MossModularPipeline.__name__,
            "_diffusers_version": diffusers.__version__,
            "uncond_skip_layers": list(uncond_skip_layers),
            "text_encoder": index_entry(
                "transformers", Qwen3_5Model, repo=snapshot, revision=None, subfolder="text_encoder"
            ),
            "tokenizer": index_entry(
                "transformers", AutoTokenizer, repo=snapshot, revision=None, subfolder="tokenizer"
            ),
            "transformer": index_entry(
                MossTransformer.__module__,
                MossTransformer,
                repo=str(transformer.resolve()),
                revision=None,
                subfolder="",
            ),
            "vae": index_entry("diffusers", AutoencoderKLFlux2, repo=snapshot, revision=None, subfolder="vae"),
        }
    )


def write_moss_repo(
    out: Path,
    *,
    base: Path,
    transformer: Path | None = None,
    uncond_skip_layers: tuple[int, ...] = DEMO_UNCOND_SKIP_LAYERS,
) -> ModelIndex:
    """Write a MOSS-Image repo into ``out``: the Base snapshot ``base``'s components around the DiT in ``transformer``
    (the snapshot's own when None), its unconditional pass skipping ``uncond_skip_layers``. Everything is checked
    before anything is written.

    Returns:
        The repo's model index.

    Raises:
        ValueError: If ``out`` already holds a model, ``base`` lacks a component's folder, or ``transformer`` is not a
            diffusers-format folder.
    """
    if (out / INDEX).exists():
        raise ValueError(f"{out} already holds a model; give a new directory")
    for folder, marker in BASE_FOLDERS.items():
        _require(base / folder / marker, "a MOSS-Image component folder")
    dit = base / "transformer" if transformer is None else transformer
    _require(dit / "config.json", "a diffusers-format transformer folder")
    index = moss_index(base, dit, uncond_skip_layers=uncond_skip_layers)
    out.mkdir(parents=True, exist_ok=True)
    index.write(out)
    return index


@app.command()
def main(
    base: Annotated[Path, typer.Option(help="the MOSS-Image Base snapshot: text_encoder/, tokenizer/, vae/")],
    out: Annotated[Path, typer.Option(help="the new model repo's directory")],
    transformer: Annotated[
        Path | None,
        typer.Option(help="the DiT folder to draw with, a merged LoRA or a fine-tune (default: --base's transformer/)"),
    ] = None,
) -> None:
    """Write a MOSS-Image model repo of the Base snapshot's components and a DiT."""
    try:
        write_moss_repo(out, base=base, transformer=transformer)
    except ValueError as error:  # every refusal is about the arguments: a usage error
        raise typer.BadParameter(str(error)) from error
    console.print(f"MOSS-Image, the DiT in {transformer or base / 'transformer'} -> {out}")


if __name__ == "__main__":
    app()
