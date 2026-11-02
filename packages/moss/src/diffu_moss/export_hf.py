"""diffu-moss-export-hf: a MOSS-Image model in the official Hub layout, the folder SGLang Diffusion loads.

    diffu-moss-export-hf runs/moss/eight_full --out runs/hf/eight_full_r6b_750
    diffu-moss-export-hf runs/moss_eight_full/full_r6b/checkpoints/step_000750 \\
        --base <MOSS-Image-Base snapshot> --out runs/hf/eight_full_r6b_750

The source is a MOSS model directory (``diffu-moss-init``'s: its index names the DiT and the Base snapshot), a DiT
folder (``config.json`` and safetensors, as ``diffu-moss-export-ref`` and a full fine-tune's checkpoint write it) or a
checkpoint holding one in ``transformer/``; the last two need ``--base``. The folder written is the Base snapshot's
layout: ``model_index.json`` (the official pipeline, ``MOSSImagePipeline``; every component's library ``diffusers`` or
``transformers``, so nothing names the Hub's ``pipeline_moss_image.py``), the Base snapshot's ``text_encoder/``,
``tokenizer/``, ``vae/`` and ``scheduler/``, and ``transformer/``: the DiT's weights, its ``config.json`` naming the
official class (``MOSSImageTransformer``), the rotary frequencies ``rope.bands`` the official checkpoint stores (our
DiT recomputes them, so its checkpoints lack them) and the DiT's ``export.json``, the record of whether it takes a
reference image (SGLang's MOSS plugin, ``diffu_sglang_moss``, reads it).

Nothing is copied by default: the components and the DiT's weights are symbolic links to where they are (some 27 GB
that would otherwise be written twice on one disk), and only the configs, the index and the 64-byte ``rope.bands``
shard are new files. ``--copy`` writes every file instead, for a folder to move to another machine whole.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Any

import diffusers
import torch
import typer
from rich.console import Console
from safetensors import safe_open
from safetensors.torch import save_file

from diffu_core.repo import INDEX, ModelIndex
from diffu_moss.train.records import EXPORT_RECORD, ExportRecord

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

PIPELINE_CLASS = "MOSSImagePipeline"
TRANSFORMER_CLASS = "MOSSImageTransformer"
BASE_COMPONENTS = {"text_encoder": "config.json", "tokenizer": "tokenizer_config.json", "vae": "config.json"}
SCHEDULER_CONFIG = "scheduler_config.json"
WEIGHTS = "diffusion_pytorch_model.safetensors"
WEIGHTS_INDEX = f"{WEIGHTS}.index.json"
ROPE_BANDS = "rope.bands"
ROPE_SHARD = "diffusion_pytorch_model-rope.safetensors"
# The Base snapshot's model_index.json, but every component loaded by its library (the Hub's names the scheduler and
# the DiT from its own pipeline_moss_image.py, which SGLang maps to the same classes)
MODEL_INDEX = {
    "_class_name": PIPELINE_CLASS,
    "scheduler": ["diffusers", "MOSSImageScheduler"],
    "text_encoder": ["transformers", "Qwen3_5Model"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "transformer": ["diffusers", TRANSFORMER_CLASS],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


def _require(path: Path, what: str) -> None:
    if not path.is_file():
        raise ValueError(f"{path.parent} is not {what}: it has no {path.name}")


def resolve_source(source: Path, base: Path | None) -> tuple[Path, Path]:
    """The DiT folder and the Base snapshot a source names: a MOSS model directory names both in its index; a DiT
    folder, or a checkpoint holding one in ``transformer/``, is the DiT, and ``base`` the snapshot.

    Raises:
        ValueError: If the source is none of those, or a DiT folder is given without ``base``.
    """
    if (source / INDEX).is_file():
        index = ModelIndex.read(source)
        transformer, vae = index.source("transformer"), index.source("vae")
        dit = Path(transformer.repo) / transformer.subfolder
        return dit, Path(vae.repo) if base is None else base
    dit = source / "transformer" if (source / "transformer" / "config.json").is_file() else source
    _require(dit / "config.json", "a MOSS model directory, a DiT folder or a checkpoint holding one")
    if base is None:
        raise ValueError(
            f"{source} is a DiT folder: give --base, the MOSS-Image Base snapshot its components come from"
        )
    return dit, base


def rope_bands(config: dict[str, Any]) -> torch.Tensor:
    """The rotary frequencies the official checkpoint stores as ``rope.bands``: ``head_dim / 8`` of them, ``1 /
    rope_base ** (k / bands)``, computed in fp32 on the CPU as the official DiT and ours compute them."""
    bands = config["dim"] // config["num_heads"] // 8
    return 1 / (config["rope_base"] ** (torch.arange(bands, dtype=torch.float32) / bands))


def _weight_map(dit: Path) -> dict[str, str]:
    """Each tensor of the DiT folder ``dit`` and the file holding it: its index's, or the single file's keys."""
    index = dit / WEIGHTS_INDEX
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    _require(dit / WEIGHTS, "a diffusers-format DiT folder")
    with safe_open(str(dit / WEIGHTS), "pt") as weights:
        return dict.fromkeys(weights.keys(), WEIGHTS)


def _place(source: Path, target: Path, *, copy: bool) -> None:
    """``target`` made a link to ``source``, or a copy of it (a folder whole, its links followed)."""
    if not copy:
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copyfile(source, target)


def export_hf(source: Path, out: Path, *, base: Path | None = None, copy: bool = False) -> bool:
    """Write the MOSS-Image model ``source`` names (with ``base``'s components) into ``out`` in the official layout.
    Everything is checked before anything is written.

    Returns:
        Whether the DiT takes a reference image, by its export's record.

    Raises:
        ValueError: If ``out`` exists, the source is not a MOSS model (:func:`resolve_source`), or the Base snapshot
            lacks a component's folder.
    """
    if out.exists():
        raise ValueError(f"{out} exists; give a new directory")
    dit, base = resolve_source(source, base)
    for folder, marker in {**BASE_COMPONENTS, "scheduler": SCHEDULER_CONFIG}.items():
        _require(base / folder / marker, "a MOSS-Image Base snapshot")
    config = json.loads((dit / "config.json").read_text(encoding="utf-8"))
    weight_map = _weight_map(dit)
    record = dit / EXPORT_RECORD
    takes_reference = record.is_file() and ExportRecord.read(record).takes_reference

    transformer = out / "transformer"
    transformer.mkdir(parents=True)
    index = {**MODEL_INDEX, "_diffusers_version": diffusers.__version__}
    (out / "model_index.json").write_text(f"{json.dumps(index, indent=2)}\n", encoding="utf-8")
    for folder in (*BASE_COMPONENTS, "scheduler"):
        _place(base / folder, out / folder, copy=copy)
    for name in sorted(set(weight_map.values())):
        _place(dit / name, transformer / name, copy=copy)
    if ROPE_BANDS not in weight_map:
        save_file({ROPE_BANDS: rope_bands(config)}, str(transformer / ROPE_SHARD), metadata={"format": "pt"})
        weight_map = {**weight_map, ROPE_BANDS: ROPE_SHARD}
    shards = {"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}
    (transformer / WEIGHTS_INDEX).write_text(f"{json.dumps(shards, indent=2)}\n", encoding="utf-8")
    config = {key: value for key, value in config.items() if key != "_name_or_path"}
    (transformer / "config.json").write_text(
        f"{json.dumps({**config, '_class_name': TRANSFORMER_CLASS}, indent=2)}\n", encoding="utf-8"
    )
    if record.is_file():
        shutil.copyfile(record, transformer / EXPORT_RECORD)
    return takes_reference


@app.command()
def main(
    source: Annotated[
        Path, typer.Argument(help="a MOSS model directory (diffu-moss-init's), a DiT folder or a checkpoint")
    ],
    out: Annotated[Path, typer.Option(help="the new folder, in the official layout")],
    base: Annotated[
        Path | None,
        typer.Option(help="the MOSS-Image Base snapshot (default: the one a model directory's index names)"),
    ] = None,
    copy: Annotated[bool, typer.Option(help="copy every file instead of linking the components and weights")] = False,
) -> None:
    """Write a MOSS-Image model in the official Hub layout, for SGLang Diffusion."""
    try:
        takes_reference = export_hf(source, out, base=base, copy=copy)
    except ValueError as error:  # every refusal is about the arguments: a usage error
        raise typer.BadParameter(str(error)) from error
    reference = "takes a reference image" if takes_reference else "text only"
    console.print(f"MOSS-Image from {source} ({reference}) -> {out}")


if __name__ == "__main__":
    app()
