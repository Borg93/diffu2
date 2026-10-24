"""Option types the generation CLIs share."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import BaseModel

from diffu_page.lora import ADAPTER_CONFIG, ADAPTER_WEIGHTS


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
