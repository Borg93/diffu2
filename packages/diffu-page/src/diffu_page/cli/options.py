"""Option types the generation CLIs share."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import BaseModel


class Lora(BaseModel):
    """A NeMo LoRA checkpoint (its ``model/`` dir) under the adapter name it loads as."""

    name: str
    path: Path


def parse_lora(value: str) -> Lora:
    """``name=<dir>`` as a :class:`Lora`.

    Raises:
        typer.BadParameter: If ``value`` is not ``name=<dir>``.
    """
    name, sep, path = value.partition("=")
    if not (sep and name and path):
        raise typer.BadParameter(f"expected name=<lora model dir>, got {value!r}")
    return Lora(name=name, path=Path(path))
