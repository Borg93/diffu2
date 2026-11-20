"""A drawn page's record: what it was drawn from, enough to draw it again."""

from __future__ import annotations

from pydantic import BaseModel

from pagedemo.document import Document


class DrawRecord(BaseModel):
    """What a page was drawn from: the model, the settings, the Layout prompt the model read, and the page as the
    editor holds it (the editor opens the record as that page again)."""

    model: str
    mode: str
    steps: int
    guidance: float
    seed: int
    megapixels: float
    width: int
    height: int
    sample_zoom: float
    prompt: str
    page: Document
