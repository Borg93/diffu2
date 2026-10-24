"""Output for the workspace's CLIs: one console for results, one on stderr for errors, and the check that a name
which output files are named after cannot point anywhere else."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()
err_console = Console(stderr=True, style="bold red")


def plain_file_name(name: str) -> str:
    """``name``, checked to be a plain file name: output files are named after it, so it must not be a path.

    Raises:
        ValueError: If ``name`` is empty, ``.``, ``..`` or holds a path separator.
    """
    if name in ("", ".", "..") or Path(name).name != name:
        raise ValueError(f"expected a plain file name, not a path: {name!r}")
    return name
