"""A MOSS-Image model repo: what the page pipeline draws with (``diffu-generate --model <out>``) and
``MossModularPipeline.from_pretrained`` loads.

    diffu-moss-init --base <MOSS-Image-Base snapshot> --out runs/moss/base
    diffu-moss-init --base <MOSS-Image-Base snapshot> --transformer <export> --out runs/moss/hand_1715

Writes ``<out>/modular_model_index.json``: the pipeline and its blocks (``MossAutoBlocks``), the DiT blocks the
unconditional pass of guidance skips, and where each component loads from: the text encoder (Qwen3.5-9B), its
tokenizer and the VAE from the Base snapshot's folders, the DiT from ``--transformer`` (a diffusers-format folder, its
``config.json`` and safetensors: a LoRA merged into Base, or a full fine-tune, as KohakuMossTrainer exports them and
as a checkpoint of ``diffu-moss-train-ref --full`` is) or the snapshot's own. Nothing is copied: the index names the
folders where they are. The scheduler is MOSS-Image's sampler at its demo's shift, 5 (``--sampling-shift`` changes it
per run).

The index's ``takes_reference`` says whether the model draws from a reference image: true only for a DiT of a
reference training (merged by ``diffu-moss-export-ref``, or a full fine-tune's checkpoint; their ``export.json`` says
so), never by hand, so a DiT that never saw a reference (the released ones, a hand LoRA) is refused one rather than
shown tokens it cannot read. Its ``sample_size``, written only for a DiT trained on samples sized otherwise than shrunk
under half a megapixel (``legible``: at the target's writing scale), is the rule the page pipeline's
``ReferenceGridStep`` sizes a sample by; an index without it sizes samples as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import diffusers
import typer
from diffusers import AutoencoderKLFlux2
from rich.console import Console
from transformers import AutoTokenizer, Qwen3_5Model

from diffu_core.repo import INDEX, ModelIndex, index_entry
from diffu_moss.modular_pipelines.moss import MossAutoBlocks, MossModularPipeline
from diffu_moss.modular_pipelines.moss.denoise import DEMO_UNCOND_SKIP_LAYERS
from diffu_moss.train.pairs import SampleSize
from diffu_moss.train.records import EXPORT_RECORD, ExportRecord
from diffu_moss.transformer import MossTransformer

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

BASE_FOLDERS = {"text_encoder": "config.json", "tokenizer": "tokenizer_config.json", "vae": "config.json"}


def _require(path: Path, what: str) -> None:
    if not path.is_file():
        raise ValueError(f"{path.parent} is not {what}: it has no {path.name}")


def export_record(transformer: Path) -> ExportRecord | None:
    """The export's record of the DiT folder ``transformer``: None for a DiT no reference training wrote."""
    record = transformer / EXPORT_RECORD
    return ExportRecord.read(record) if record.is_file() else None


def takes_reference(transformer: Path) -> bool:
    """Whether the DiT folder ``transformer`` was trained to take a reference image, by its export's record."""
    record = export_record(transformer)
    return record is not None and record.takes_reference


def sizing(transformer: Path) -> dict[str, str]:
    """The model index's entry of how the DiT folder ``transformer`` was trained to read its samples sized, by its
    export's record: none for samples shrunk under half a megapixel (or no record), as an index was written before."""
    record = export_record(transformer)
    if record is None or record.sample_size is SampleSize.HALF:
        return {}
    return {"sample_size": record.sample_size.value}


def moss_index(base: Path, transformer: Path, *, uncond_skip_layers: tuple[int, ...]) -> ModelIndex:
    """The ``modular_model_index.json`` of a MOSS-Image repo: its pipeline and block classes, the blocks its
    unconditional pass skips, whether it takes a reference image and how its samples are sized (:func:`sizing`), the
    text encoder, tokenizer and VAE in the Base snapshot ``base``'s folders and the DiT in the folder ``transformer``."""
    snapshot = str(base.resolve())
    return ModelIndex.model_validate(
        {
            "_blocks_class_name": MossAutoBlocks.__name__,
            "_class_name": MossModularPipeline.__name__,
            "_diffusers_version": diffusers.__version__,
            "uncond_skip_layers": list(uncond_skip_layers),
            "takes_reference": takes_reference(transformer),
            **sizing(transformer),
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
        index = write_moss_repo(out, base=base, transformer=transformer)
    except ValueError as error:  # every refusal is about the arguments: a usage error
        raise typer.BadParameter(str(error)) from error
    extra = index.model_extra or {}
    sized = extra.get("sample_size", SampleSize.HALF.value)
    reference = f"takes a reference image sized {sized}" if extra.get("takes_reference") else "text only"
    console.print(f"MOSS-Image, the DiT in {transformer or base / 'transformer'} ({reference}) -> {out}")


if __name__ == "__main__":
    app()
