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

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, Flux3Transformer2DModel
from rich.console import Console

from diffu.init import INDEX, index_entry

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()
err_console = Console(stderr=True)


def trained_shift(checkpoint: Path) -> float:
    """The timestep shift ``checkpoint`` trained at: its ``config.yaml``'s ``flow_matching.flow_shift``.

    Raises:
        ValueError: If the config names none: NeMo would have trained at its own default.
    """
    config = yaml.safe_load((checkpoint / "config.yaml").read_text(encoding="utf-8"))
    shift = config.get("flow_matching", {}).get("flow_shift")
    if shift is None:
        raise ValueError(f"{checkpoint}/config.yaml names no flow_matching.flow_shift")
    return float(shift)


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
    if (out / INDEX).exists() or (out / "transformer").exists():
        raise ValueError(f"{out} already holds a model; give a new directory")
    transformer = checkpoint / "model" / "consolidated"
    if not (transformer / "config.json").is_file():
        raise ValueError(f"{checkpoint} has no consolidated export: export the run's last checkpoint")
    shift = trained_shift(checkpoint)
    index = json.loads((init / INDEX).read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "transformer").symlink_to(transformer.resolve(), target_is_directory=True)
    FlowMatchEulerDiscreteScheduler(shift=shift).save_pretrained(out / "scheduler")
    repo = str(out.resolve())
    index["transformer"] = index_entry(
        "diffusers", Flux3Transformer2DModel, repo=repo, revision=None, subfolder="transformer"
    )
    index["scheduler"] = index_entry(
        "diffusers", FlowMatchEulerDiscreteScheduler, repo=repo, revision=None, subfolder="scheduler"
    )
    (out / INDEX).write_text(f"{json.dumps(index, indent=2)}\n", encoding="utf-8")
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
    except ValueError as error:
        err_console.print(str(error))
        raise typer.Exit(code=1) from error
    console.print(f"{checkpoint.name} at shift {shift:g} -> {out}")


if __name__ == "__main__":
    app()
