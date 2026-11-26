"""Option types the generation CLIs share: the draw options, each declared once with its default from
``DrawSettings`` and mapped back onto its field by :func:`settings_from_options`, and the LoRA and model options."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from diffu_page.draw.lora import lora_model_dir
from diffu_page.draw.pipeline import DrawSettings
from diffu_page.output import validation_problems
from diffu_page.page.labels import LabelFormat

DEFAULTS = DrawSettings()  # every draw option's default comes from here

Steps = Annotated[int, typer.Option(min=1, help="denoising steps per crop")]
Guidance = Annotated[
    float | None,
    typer.Option(
        help="how hard guidance pushes toward the caption (default 2: closer to a sample page's ink than MOSS-Image's own 3)"
    ),
]
ReferenceGuidanceScale = Annotated[
    float, typer.Option(help="how hard guidance also pushes toward the reference image (1: not at all)")
]
GuidanceStop = Annotated[
    float, typer.Option(min=0.0, max=1.0, help="the fraction of the steps after which guidance stops (1: every step)")
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
RepairPasses = Annotated[
    int,
    typer.Option(
        "--repair", min=0, help="times a crop's lines the read-back rejects are drawn again in place (0: none)"
    ),
]


class Lora(BaseModel):
    """A LoRA adapter folder (``diffu-moss-convert-lora``'s, of MOSS-Image's DiT) under the adapter name it loads as."""

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
    """``name=<dir>`` as a :class:`Lora`, its dir checked to be a LoRA adapter folder.

    Raises:
        typer.BadParameter: If ``value`` is not ``name=<dir>``, or the dir lacks an adapter folder's files.
    """
    name, path = named_path(value, "name=<lora adapter folder>")
    try:
        lora_model_dir(path)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    return Lora(name=name, path=path)


Model = Annotated[
    Path | None, typer.Option(help="the MOSS-Image model directory (diffu-moss-init's); default $DIFFU_MOSS_MODEL")
]
Loras = Annotated[
    list[Lora] | None,
    typer.Option(
        parser=parse_lora,
        help="name=<LoRA adapter folder>: diffu-moss-convert-lora's, of MOSS-Image's DiT",
    ),
]


def settings_from_options(
    *,
    steps: int,
    guidance: float | None,
    reference_guidance: float,
    guidance_stop: float,
    device: str,
    max_line_cer: float,
    lora_scale: float,
    sampling_shift: float | None,
    fuse_loras: bool,
    compile_blocks: bool,
    label_format: LabelFormat = DEFAULTS.label_format,
    repair_passes: int = DEFAULTS.repair_passes,
) -> DrawSettings:
    """The draw options a command was given, by their parameters' names, as ``DrawSettings``. Each is required, so a
    command that leaves one out fails the type check; the label format and the repair passes matter only to a command
    that writes labels and keeps its crops."""
    return DrawSettings(
        steps=steps,
        guidance_scale=guidance,
        reference_guidance_scale=reference_guidance,
        guidance_stop=guidance_stop,
        label_format=label_format,
        max_line_cer=max_line_cer,
        device=device,
        lora_scale=lora_scale,
        sampling_shift=sampling_shift,
        fuse_loras=fuse_loras,
        compile_blocks=compile_blocks,
        repair_passes=repair_passes,
    )


DrawSettingsFile = Annotated[
    Path | None,
    typer.Option(
        "--draw-settings",
        help="a JSON object of draw settings (DrawSettings fields by name, the speed options among them: sampler, "
        "cache, precision, text_encoder_precision, aoti_package) whose values replace the flags'",
    ),
]


def with_settings_file(flags: DrawSettings, path: Path | None) -> DrawSettings:
    """The settings the flags gave, with each field the JSON object at ``path`` names set to its value there; the
    flags' alone without a file.

    Raises:
        typer.BadParameter: If the file is not a JSON object of draw settings, or names a key that is not one.
    """
    if path is None:
        return flags
    hint = "--draw-settings"
    try:
        given = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"{path} is not a JSON object of draw settings: {error}", param_hint=hint) from error
    if not isinstance(given, dict):
        raise typer.BadParameter(f"{path} is not a JSON object of draw settings", param_hint=hint)
    unknown = sorted(set(given) - set(DrawSettings.model_fields))
    if unknown:
        raise typer.BadParameter(f"{path} names draw settings that do not exist: {', '.join(unknown)}", param_hint=hint)
    try:
        return DrawSettings.model_validate(flags.model_dump() | given)
    except ValidationError as error:
        raise typer.BadParameter(f"{path}: {validation_problems(error, 'the settings')}", param_hint=hint) from error


def lora_dirs(loras: list[Lora] | None) -> dict[str, Path]:
    """The LoRA adapter folders by adapter name.

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
    """The model directory a command draws with: ``--model``, else ``DIFFU_MOSS_MODEL``.

    Raises:
        typer.BadParameter: If neither names one.
    """
    if model is None:
        raise typer.BadParameter("set DIFFU_MOSS_MODEL (in the environment or a .env file) or pass --model")
    return model
