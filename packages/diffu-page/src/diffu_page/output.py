"""Terminal output for the workspace's CLIs: one console for results, one on stderr for errors."""

from __future__ import annotations

from rich.console import Console

console = Console()
err_console = Console(stderr=True, style="bold red")
