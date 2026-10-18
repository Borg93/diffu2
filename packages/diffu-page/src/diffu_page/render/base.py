"""Renderer seam + the no-model placeholder.

:class:`LineRenderer` is the Protocol every line generator satisfies; :class:`LineJob` is one line's
work order for the batched path; :class:`PlaceholderRenderer` draws gray OpenCV text so the layout +
compositor are runnable and testable with no model, no GPU, no torch.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
from pydantic import BaseModel


class LineJob(BaseModel):
    """One line to render: its text, the target ink height (px), and the writer-style reference path."""

    model_config = {"frozen": True}

    text: str
    height: int
    style_path: str


class LineRenderer(Protocol):
    """Renders one text line to a BGRA ink raster at approximately ``height`` pixels tall."""

    def render(self, text: str, height: int) -> np.ndarray: ...


class PlaceholderRenderer:
    """Gray Hershey-font text → BGRA ink. For developing/testing the page pipeline without a model."""

    def __init__(self, font: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
        self.font = font

    def render(self, text: str, height: int) -> np.ndarray:
        text = text or " "
        thickness = max(1, height // 22)
        scale = cv2.getFontScaleFromHeight(self.font, max(8, int(height * 0.62)), thickness)
        (tw, th), _ = cv2.getTextSize(text, self.font, scale, thickness)
        pad = max(4, int(height * 0.25))
        canvas = np.zeros((height + 2 * pad, tw + 2 * pad), np.uint8)  # alpha/coverage
        baseline_y = pad + th
        cv2.putText(canvas, text, (pad, baseline_y), self.font, scale, 255, thickness, cv2.LINE_AA)
        bgra = np.zeros((canvas.shape[0], canvas.shape[1], 4), np.uint8)
        bgra[..., 3] = canvas
        return bgra

    def measure_width(self, text: str, height: int) -> int:
        """Predicted rendered width of ``text`` at ``height`` — matches :meth:`render` exactly."""
        text = text or " "
        thickness = max(1, height // 22)
        scale = cv2.getFontScaleFromHeight(self.font, max(8, int(height * 0.62)), thickness)
        (tw, _), _ = cv2.getTextSize(text, self.font, scale, thickness)
        return tw + 2 * max(4, int(height * 0.25))
