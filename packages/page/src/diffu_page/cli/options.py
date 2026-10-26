"""Option types the generation CLIs share: the draw options, each declared once with its default from
``DrawSettings`` and mapped back onto its field by :func:`settings_from_options`, and the LoRA and model options."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from diffu_page.labels import LabelFormat
from diffu_page.lora import ADAPTER_CONFIG, ADAPTER_WEIGHTS
from diffu_page.pipeline import DrawSettings

DEFAULTS = DrawSettings()  # every draw option's default comes from here

Steps = Annotated[int, typer.Option(min=1, help="denoising steps per crop")]
Guidance = Annotated[float, typer.Option(help="how hard guidance pushes toward the caption")]
ReferenceGuidanceScale = Annotated[
    float, typer.Option(help="how hard guidance also pushes toward the reference image (1: not at all)")
]
GuidanceStop = Annotated[
    float, typer.Option(min=0.0, max=1.0, help="the fraction of the steps after which guidance stops (1: every step)")
]
TextLayers = Annotated[
    list[int] | None,
    typer.Option("--text-layer", help="a text-encoder layer to stack (repeat it; default: the model's own)"),
]
Device = Annotated[str, typer.Option(help="where the models run")]
MaxLineCer = Annotated[float, typer.Option(help="the read-back CER above which a line's label is tagged unclear")]
Format = Annotated[LabelFormat, typer.Option("--format", help="the labels' format")]
SamplingShift = Annotated[
    float | None, typer.Option(min=0.0, help="the timestep shift to sample at (default: the model's own)")
]
LoraScale = Annotated[
    float, typer.Option(min=0.0, help="how strongly the LoRAs apply: 1 at their trained strength, 0 not at all")
]
FuseLoras = Annotated[
    bool, typer.Option(help="fuse the LoRAs into the transformer at load: faster, but they can't be switched after")
]
CompileBlocks = Annotated[
    bool, typer.Option(help="compile the transformer's repeated blocks at load (the first crop pays); fuse LoRAs first")
]


class Lora(BaseModel):
    """A NeMo LoRA checkpoint (its ``model/`` dir) under the adapter name it loads as."""

    name: str
    path: Path


def named_path(value: str, form: str) -> tuple[str, Path]:
    """``<name>=<path>`` split into its name and path.

    Raises:
        typer.BadParameter: If ``value`` is not of that form; ``form`` names the expected one in the message.
    """
    name, sep, path = value.partition("=")
    if not (sep and name and path):
        raise typer.BadParameter(f"expected {form}, got {value!r}")
    return name, Path(path)


def parse_lora(value: str) -> Lora:
    """``name=<dir>`` as a :class:`Lora`, its dir checked to be a NeMo LoRA checkpoint's ``model/`` dir.

    Raises:
        typer.BadParameter: If ``value`` is not ``name=<dir>``, or the dir lacks a LoRA checkpoint's files.
    """
    name, path = named_path(value, "name=<lora model dir>")
    missing = [file for file in (ADAPTER_CONFIG, ADAPTER_WEIGHTS) if not (path / file).is_file()]
    if missing:
        raise typer.BadParameter(f"{path} is not a NeMo LoRA model dir: it has no {' or '.join(missing)}")
    return Lora(name=name, path=path)


Model = Annotated[Path | None, typer.Option(help="the model directory (FLUX.3 Klein or diffu); default $FLUX3_MODEL")]
Loras = Annotated[list[Lora] | None, typer.Option(parser=parse_lora, help="name=<NeMo LoRA model dir>")]


def text_layers(layers: list[int] | None) -> tuple[int, ...] | None:
    """The ``--text-layer`` options as ``DrawSettings`` takes them: None (the model's own layers) when none is given."""
    return tuple(layers) if layers else None


def settings_from_options(
    *,
    steps: int,
    guidance: float,
    reference_guidance: float,
    guidance_stop: float,
    text_layer: list[int] | None,
    device: str,
    max_line_cer: float,
    lora_scale: float,
    sampling_shift: float | None,
    fuse_loras: bool,
    compile_blocks: bool,
    label_format: LabelFormat = DEFAULTS.label_format,
) -> DrawSettings:
    """The draw options a command was given, by their parameters' names, as ``DrawSettings``. Each is required, so a
    command that leaves one out fails the type check; the label format matters only to a command that writes labels."""
    return DrawSettings(
        steps=steps,
        guidance_scale=guidance,
        reference_guidance_scale=reference_guidance,
        guidance_stop=guidance_stop,
        label_format=label_format,
        max_line_cer=max_line_cer,
        text_layers=text_layers(text_layer),
        device=device,
        lora_scale=lora_scale,
        sampling_shift=sampling_shift,
        fuse_loras=fuse_loras,
        compile_blocks=compile_blocks,
    )


def lora_dirs(loras: list[Lora] | None) -> dict[str, Path]:
    """The LoRA model dirs by adapter name.

    Raises:
        typer.BadParameter: If two LoRAs share a name: the later would silently replace the earlier.
    """
    dirs: dict[str, Path] = {}
    for lora in loras or []:
        if lora.name in dirs:
            raise typer.BadParameter(
                f"two LoRAs are named {lora.name!r}: {dirs[lora.name]} and {lora.path}", param_hint="--lora"
            )
        dirs[lora.name] = lora.path
    return dirs


def required_model(model: Path | None) -> Path:
    """The model directory a command draws with: ``--model``, else ``FLUX3_MODEL``.

    Raises:
        typer.BadParameter: If neither names one.
    """
    if model is None:
        raise typer.BadParameter("set FLUX3_MODEL (in the environment or a .env file) or pass --model")
    return model
