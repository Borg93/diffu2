"""The values streamed out of the core generators — one per animation frame / seated line.

Deliberately NOT ``from __future__ import annotations``: these models carry live ``PIL.Image`` objects and
Pydantic needs the real ``Image`` type (not a string) to build the model with ``arbitrary_types_allowed``.
The UI (``ui.py``) and the API (``server.py``) each adapt these to their own output shape.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from diffu_page.config import GenConfig
from diffu_page.labels import PageLabels
from diffu_page.layout import Layout
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict


class LineStep(BaseModel):
    """One frame of single-line generation: a denoise still, or the final ink-cropped line with its readback."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image | None
    status: str
    index: int
    total: int
    done: bool = False
    read: str | None = None  # TrOCR read-back + CER, on the final frame only
    error: str | None = None


class RenderCache(BaseModel):
    """The model's output for a page — the layout + one BGRA ink per box — kept so scan-look AND cheap layout
    changes re-composite on the CPU without re-running the (GPU) denoise. Carried on the final frame.

    ``gen``/``paragraphs``/``xml_layout``/``measure`` are what's needed to REBUILD the layout at new geometry
    (line/paragraph gap) with the same wrapping — so the cached inks re-map to the new boxes 1:1. ``measure``
    is the model's width predictor: without it the rebuild would wrap differently and the reseat would (safely)
    fall back to a full re-generate."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    layout: Layout
    inks: list[np.ndarray]  # per-box BGRA arrays, aligned to layout.boxes
    seed: int
    gen: GenConfig
    paragraphs: list[str]
    xml_layout: str | None = None
    measure: Callable[[str, int], int] | None = None

    def __getstate__(self) -> dict:
        """Drop ``measure`` when pickling: it is a closure over the loaded model, and on ZeroGPU this cache
        crosses the fork→parent boundary via pickle after Generate. Without it, a later spacing change falls
        back safely (crude estimate or a "re-generate" notice) instead of crashing the whole Generate."""
        state = super().__getstate__()
        inner = state.get("__dict__")  # pydantic v2 keeps field values here
        if isinstance(inner, dict) and inner.get("measure") is not None:
            state = {**state, "__dict__": {**inner, "measure": None}}
        return state


class PageStep(BaseModel):
    """One frame of whole-page generation: the page as it grows, plus (while animating) the active line."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page: Image | None
    active: Image | None = None  # the line currently denoising (preview), None once seated
    status: str
    index: int  # lines seated so far
    total: int
    phase: Literal["denoise", "seated", "final", "error"]
    labels: dict[str, str] | None = None  # {format: serialized text}, on the final frame only
    page_labels: PageLabels | None = None  # structured GT (polygons+text) for the overlay, on the final frame
    render: RenderCache | None = (
        None  # cached layout + inks, so augmentation changes re-composite w/o the model
    )
    error: str | None = None
