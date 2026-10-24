"""The empty prompt written into a diffu cache made before caches kept it (``diffu_train.cache_diffu`` writes it with
every cache now), so caption dropout can train on it.

    python -m diffu_train.empty_prompt --model <the diffu repo the cache was made with> --cache <cache dir>
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from diffu_page.output import console
from diffu_train.cache import DiffuEncoders, write_empty_prompt
from diffu_train.dropout import EMPTY_PROMPT_FILE

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def main(
    model: Annotated[Path, typer.Option(help="the diffu repo whose encoders made the cache")],
    cache: Annotated[Path, typer.Option(help="a finished diffu cache (diffu_train.cache_diffu)")],
    device: Annotated[str, typer.Option(help="where the text encoder runs")] = "cuda",
) -> None:
    """Write the empty prompt, encoded as diffu's pipeline encodes a caption, into an existing cache."""
    if not (cache / "metadata.json").is_file():
        raise typer.BadParameter(f"{cache} holds no finished cache (no metadata.json)", param_hint="--cache")
    write_empty_prompt(DiffuEncoders.load(model, device), cache)
    console.print(f"the empty prompt -> {cache / EMPTY_PROMPT_FILE}")


if __name__ == "__main__":
    app()
