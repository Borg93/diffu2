"""Output for htr-train's CLI: one console for results, one on stderr for errors, and how a validation error reads in a
message (as ``diffu_page.output`` words it; htr-train does not import diffu-page)."""

from __future__ import annotations

from pydantic import ValidationError
from rich.console import Console

console = Console()
err_console = Console(stderr=True, style="bold red")


def validation_problems(error: ValidationError, whole: str) -> str:
    """What ``error`` found wrong, for a message: each problem as ``<field path>: <what>`` (``whole`` when it is the
    input itself), joined by ``; ``."""
    return "; ".join(f"{'.'.join(map(str, problem['loc'])) or whole}: {problem['msg']}" for problem in error.errors())
