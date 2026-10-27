"""Printed or handwritten: a small CNN that reads a text line's image (:mod:`diffu_page.marking.lines`) window by
window, and the classifier that holds it and gives each line the chance that it is printed. The weights are trained by
:mod:`diffu_page.marking.train` and live outside git (``runs/print_hand/``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn

WINDOW = 256  # px: the width of the windows a line image is read in
STRIDE = 128  # px between a line's windows
MAX_WINDOWS = 8  # a longer line is read in this many windows, spread over it
CHANNELS = (1, 16, 32, 64, 96)
PRINTED_CHANCE = 0.9  # a line read as printed with less chance than this is uncertain, and stays handwritten


class PrintHandNet(nn.Module):
    """Four 3x3 convolution blocks (batch norm, ReLU, 2x2 max-pool) widening to 96 channels, global average pooling and
    one logit: printed. It reads windows of ink, ``N x 1 x 64 x width``, paper 0 and ink 1."""

    def __init__(self) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for c_in, c_out in pairwise(CHANNELS):
            blocks += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(CHANNELS[-1], 1))

    def forward(  # ty: ignore[invalid-mutable-override]  torch declares forward an attribute; Modules override it
        self, windows: torch.Tensor
    ) -> torch.Tensor:
        """The printed logit of each window."""
        return self.head(self.features(windows).mean(dim=(2, 3))).squeeze(1)


def ink(image: np.ndarray) -> np.ndarray:
    """A line image (uint8, ink dark on white) as the net reads it: float32, paper 0 and ink 1."""
    return 1.0 - image.astype(np.float32) / 255.0


def windows(image: np.ndarray) -> np.ndarray:
    """A line image's windows as the net reads them (``N x 1 x 64 x WINDOW``, :func:`ink`): :data:`WINDOW` px wide and
    :data:`STRIDE` apart, at most :data:`MAX_WINDOWS` spread evenly over the line, a shorter line padded with paper."""
    line = ink(image)
    width = line.shape[1]
    if width < WINDOW:
        line = np.pad(line, ((0, 0), (0, WINDOW - width)))
        width = WINDOW
    count = min(MAX_WINDOWS, 1 + math.ceil((width - WINDOW) / STRIDE))
    starts = np.linspace(0, width - WINDOW, count).round().astype(int)
    return np.stack([line[:, start : start + WINDOW] for start in starts])[:, None]


class PrintHandClassifier:
    """Reads text lines as printed or handwritten with a :class:`PrintHandNet` (or any module of its input and output),
    on CPU. Load it once (a stage's actor does in its ``__init__``) and call it with each line's views."""

    def __init__(self, net: nn.Module) -> None:
        self.net = net.eval()

    @classmethod
    def load(cls, weights: Path) -> PrintHandClassifier:
        """A classifier of the :class:`PrintHandNet` weights :mod:`diffu_page.marking.train` saved at ``weights``."""
        net = PrintHandNet()
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        return cls(net)

    @torch.inference_mode()
    def __call__(self, lines: Sequence[Sequence[np.ndarray]]) -> list[float]:
        """The chance that each line is printed: the mean over every window of its views
        (:func:`diffu_page.marking.lines.line_images`); a line with no view gets 0, handwritten."""
        chances: list[float] = []
        for views in lines:
            if not views:
                chances.append(0.0)
                continue
            batch = torch.from_numpy(np.concatenate([windows(view) for view in views]))
            chances.append(float(torch.sigmoid(self.net(batch)).mean()))
        return chances
