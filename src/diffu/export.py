"""A trained diffu model repo from a NeMo AutoModel checkpoint: the init repo's components around the trained
transformer, sampled at the timestep shift it trained at.

    diffu-export --checkpoint runs/diffu_s_smoke/epoch_3_step_1999 --init runs/diffu_s_smoke/init \\
        --out runs/diffu_s_smoke/model

Writes ``<out>/transformer``, a link to the checkpoint's consolidated diffusers export; ``<out>/scheduler/``, FLUX.3's
sampler at the checkpoint's ``flow_matching.flow_shift`` (without it, FLUX.3's blocks sample at FLUX.3's own 6.93);
and, last, ``<out>/modular_model_index.json``: the init repo's, with the transformer and the scheduler loading from
``<out>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel
from pydantic import BaseModel, ValidationError
from rich.console import Console

from diffu.repo import INDEX, ModelIndex, index_entry

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()


class FlowMatching(BaseModel):
    """The flow-matching section of a NeMo run's config: the timestep shift it trained at."""

    flow_shift: float


class RunConfig(BaseModel):
    """The part of a NeMo run's ``config.yaml`` an export needs; the rest is kept out."""

    flow_matching: FlowMatching


def trained_shift(checkpoint: Path) -> float:
    """The timestep shift ``checkpoint`` trained at: its ``config.yaml``'s ``flow_matching.flow_shift``.

    Raises:
        ValueError: If the config names none (NeMo would have trained at its own default) or is not a mapping.
    """
    path = checkpoint / "config.yaml"
    try:
        return RunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))).flow_matching.flow_shift
    except ValidationError as error:
        raise ValueError(f"{path} names no flow_matching.flow_shift") from error


def export(checkpoint: Path, init: Path, out: Path) -> float:
    """Write ``out``, a diffu model repo of ``checkpoint``'s transformer and ``init``'s other components, sampled at the
    shift the checkpoint trained at. The checkpoint is checked before anything is written.

    Args:
        checkpoint: A NeMo checkpoint, ``epoch_*_step_*``.
        init: The diffu repo the run trained from (diffu-init's).
        out: The new repo's directory.

    Returns:
        The shift.

    Raises:
        ValueError: If ``out`` already holds a model, or the checkpoint has no consolidated export (under
            ``save_consolidated: final``, NeMo writes one for the last checkpoint only) or its config names no flow
            shift.
    """
    if (out / INDEX).exists() or (out / "transformer").is_symlink() or (out / "transformer").exists():
        raise ValueError(f"{out} already holds a model; give a new directory")
    transformer = checkpoint / "model" / "consolidated"
    if not (transformer / "config.json").is_file():
        raise ValueError(f"{checkpoint} has no consolidated export: export the run's last checkpoint")
    shift = trained_shift(checkpoint)
    index = ModelIndex.read(init)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transformer").symlink_to(transformer.resolve(), target_is_directory=True)
    FlowMatchEulerDiscreteScheduler(shift=shift).save_pretrained(out / "scheduler")
    repo = str(out.resolve())
    trained = {
        "transformer": index_entry(
            "diffusers", Flux3Transformer2DModel, repo=repo, revision=None, subfolder="transformer"
        ),
        "scheduler": index_entry(
            "diffusers", FlowMatchEulerDiscreteScheduler, repo=repo, revision=None, subfolder="scheduler"
        ),
    }
    index.with_components(trained).write(out)
    return shift


@app.command()
def main(
    checkpoint: Annotated[Path, typer.Option(help="the trained checkpoint, the run's last epoch_*_step_* dir")],
    init: Annotated[Path, typer.Option(help="the diffu repo the run trained from (diffu-init's)")],
    out: Annotated[Path, typer.Option(help="the new model repo's directory")],
) -> None:
    """Write a diffu model repo of a trained checkpoint, sampled at the timestep shift it trained at."""
    try:
        shift = export(checkpoint, init, out)
    except ValueError as error:  # every refusal is about the arguments: a usage error
        raise typer.BadParameter(str(error)) from error
    console.print(f"{checkpoint.name} at shift {shift:g} -> {out}")


if __name__ == "__main__":
    app()
