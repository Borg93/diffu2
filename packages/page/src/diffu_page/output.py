"""Output for the workspace's CLIs: one console for results, one on stderr for errors, the check that a name which
output files are named after cannot point anywhere else, and how images are read and written."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from rich.console import Console

console = Console()
err_console = Console(stderr=True, style="bold red")
JPEG_QUALITY = 92  # drawn crops, sheets and references (training tiles: build_crops' TILE_QUALITY)


def jpeg_bytes(image: Image.Image) -> bytes:
    """``image`` as JPEG at ``JPEG_QUALITY``."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def side_by_side(panels: list[Image.Image], *, height: int, gap: int, top: int = 0) -> Image.Image:
    """The panels scaled to ``height`` and pasted left to right ``gap`` px apart on white, ``top`` px down (room for a
    label above them)."""
    scaled = [panel.convert("RGB").resize((round(panel.width * height / panel.height), height)) for panel in panels]
    sheet = Image.new("RGB", (sum(p.width for p in scaled) + gap * (len(scaled) - 1), top + height), "white")
    x = 0
    for panel in scaled:
        sheet.paste(panel, (x, top))
        x += panel.width + gap
    return sheet


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
