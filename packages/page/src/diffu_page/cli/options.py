"""Option types the generation CLIs share: the draw options, each declared once with its default from
``DrawSettings``, and the LoRA and model options."""

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
ReferenceGuidance = Annotated[
    float, typer.Option(help="how hard guidance also pushes toward the reference image (1: not at all)")
]
TextLayers = Annotated[
    list[int] | None,
    typer.Option("--text-layer", help="a text-encoder layer to stack (repeat it; default: the model's own)"),
]
Device = Annotated[str, typer.Option(help="where the models run")]
MaxLineCer = Annotated[float, typer.Option(help="the read-back CER above which a line's label is tagged unclear")]
Format = Annotated[LabelFormat, typer.Option("--format", help="the labels' format")]


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


def required_model(model: Path | None) -> Path:
    """The model directory a command draws with: ``--model``, else ``FLUX3_MODEL``.

    Raises:
        typer.BadParameter: If neither names one.
    """
    if model is None:
        raise typer.BadParameter("set FLUX3_MODEL (in the environment or a .env file) or pass --model")
    return model
