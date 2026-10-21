"""Fixtures shared by every workspace member's tests: a PAGE page with a body block and two margin notes, and
diffusers' tiny FLUX.3 test model."""

from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub import snapshot_download


def _line(line_id: str, x0: int, x1: int, baseline: int, text: str) -> str:
    return (
        f'<TextLine id="{line_id}"><Coords points="{x0},{baseline - 60} {x1},{baseline - 60} {x1},{baseline + 20} '
        f'{x0},{baseline + 20}"/><Baseline points="{x0},{baseline} {x1},{baseline}"/>'
        f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
    )


@pytest.fixture
def body_and_notes_page(tmp_path: Path) -> Path:
    """A 2000 x 1900 px page: 14 body lines at a 100 px pitch, and two margin notes left of body lines 3 and 4."""
    body = "".join(_line(f"b{i}", 600, 1900, 300 + 100 * i, f"body line {i} of the running text") for i in range(14))
    notes = "".join(_line(f"n{i}", 120, 520, baseline, f"note {i}") for i, baseline in enumerate((600, 700)))
    path = tmp_path / "page.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="page.jpg" imageWidth="2000" imageHeight="1900">'
        f'<TextRegion id="body" type="paragraph">{body}</TextRegion>'
        f'<TextRegion id="notes" type="marginalia">{notes}</TextRegion>'
        "</Page></PcGts>",
        encoding="utf-8",
    )
    return path


TINY_FLUX3 = "hf-internal-testing/tiny-flux3-image-modular"
TINY_FLUX3_REVISION = "9130f96f124d6c6fdd1beab63cfb6d2f1c8658a1"  # pinned: a changed fixture must not change results


@pytest.fixture(scope="session")
def tiny_model() -> Path:
    """diffusers' tiny FLUX.3 test model (a few MB), downloaded once into the Hugging Face cache."""
    return Path(snapshot_download(TINY_FLUX3, revision=TINY_FLUX3_REVISION))
