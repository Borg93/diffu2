"""Output for the workspace's CLIs: one console for results, one on stderr for errors, the check that a name which
output files are named after cannot point anywhere else, and how images are read and written."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from rich.console import Console

console = Console()
err_console = Console(stderr=True, style="bold red")
JPEG_QUALITY = 92  # every crop, sheet and reference written as JPEG


def jpeg_bytes(image: Image.Image) -> bytes:
    """``image`` as JPEG at ``JPEG_QUALITY``."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def load_rgb(path: Path) -> Image.Image:
    """The image at ``path``, read whole and as RGB (the file is closed when it returns)."""
    with Image.open(path) as image:
        return image.convert("RGB")


def plain_file_name(name: str) -> str:
    """``name``, checked to be a plain file name: output files are named after it, so it must not be a path.

    Raises:
        ValueError: If ``name`` is empty, ``.``, ``..`` or holds a path separator.
    """
    if name in ("", ".", "..") or Path(name).name != name:
        raise ValueError(f"expected a plain file name, not a path: {name!r}")
    return name
