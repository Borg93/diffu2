"""A tiny pool and ground truth on disk: ALTO files written as the old pipeline and a transcriber's export write
them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from diffu_curate.settings import Pool

type Line = tuple[str, tuple[int, int, int, int]]  # a line's text and its box as x, y, width, height

ALTO = """<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
<Description><MeasurementUnit>pixel</MeasurementUnit>
<Processing><processingSoftware><softwareName>{writer}</softwareName></processingSoftware></Processing></Description>
<Layout><Page WIDTH="{width}" HEIGHT="{height}"{confidence}><PrintSpace><TextBlock ID="b">{lines}</TextBlock></PrintSpace></Page></Layout>
</alto>"""


def _alto(
    writer: str, lines: list[Line], *, size: tuple[int, int] = (1000, 800), confidence: float | None = None
) -> str:
    """An ALTO page of ``lines``, each with its box as its polygon and its words as ``String`` elements."""
    written = []
    for text, (x, y, w, h) in lines:
        words = "".join(f'<String CONTENT="{word}"/>' for word in text.split())
        polygon = f"{x},{y} {x + w},{y} {x + w},{y + h} {x},{y + h}"
        written.append(
            f'<TextLine HPOS="{x}" VPOS="{y}" WIDTH="{w}" HEIGHT="{h}">'
            f'<Shape><Polygon POINTS="{polygon}"/></Shape>{words}</TextLine>'
        )
    pc = f' PC="{confidence}"' if confidence is not None else ""
    return ALTO.format(writer=writer, width=size[0], height=size[1], confidence=pc, lines="".join(written))


@pytest.fixture
def alto() -> Callable[..., str]:
    """Writes an ALTO page: ``alto(writer, lines, size=..., confidence=...)``."""
    return _alto


@pytest.fixture
def pool(tmp_path: Path) -> Pool:
    """An empty pool named ``pool``."""
    labels, scans = tmp_path / "labels", tmp_path / "scans"
    labels.mkdir()
    scans.mkdir()
    return Pool(name="pool", labels=labels, scans=scans)


@pytest.fixture
def add_scan(pool: Pool) -> Callable[[str, str], Path]:
    """Adds a scan to the pool: its label file's content under its stem (and an image file), returning the label
    file."""

    def _add(stem: str, content: str) -> Path:
        label = pool.label_file(stem)
        label.parent.mkdir(exist_ok=True)
        label.write_text(content, encoding="utf-8")
        image = pool.scan_file(stem)
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(b"jpg")
        return label

    return _add
