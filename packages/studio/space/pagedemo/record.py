"""A drawn page's record: what it was drawn from, enough to draw it again."""

from __future__ import annotations

from pydantic import BaseModel, Field

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
    old_swedish: bool = Field(default=False, exclude_if=lambda used: not used)  # drawn with the Old Swedish LoRA
    prompt: str
    page: Document
