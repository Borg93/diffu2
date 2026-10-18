"""Quality gate — the checks that keep a model misfire out of the dataset.

Two layers, cheapest first: a free image check (blank / smeared-doubled ink density) then a TrOCR CER
read-back (the real gauge). Pure functions here; the actors in :mod:`diffu_runner.actors` apply them. The
CER threshold turns "2.6% mean with the rare garbled line" into a clean set.
"""

from __future__ import annotations

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel


class GateConfig(BaseModel):
    """Thresholds for the quality gate (frozen)."""

    model_config = {"frozen": True}

    cer_max: float = 0.30  # drop a line the recognizer reads back worse than this
    ink_min: float = 0.004  # below this ink fraction = a blank output
    ink_max: float = 0.45  # above this = an unambiguous blob / doubled render (only the extreme; TrOCR
    #                         arbitrates dense-but-legible lines so good bold hands are kept)


def ink_fraction(img: Image, margin: int = 40) -> float:
    """Fraction of clearly-inked pixels (darker than the paper median). Blank ≈ 0, normal line ≈ 0.05–0.20,
    a doubled/smeared misfire runs much denser — so extremes flag the obvious junk before any TrOCR cost."""
    gray = np.asarray(img.convert("L"), dtype=np.int16)
    paper = int(np.median(gray))
    return float((gray < paper - margin).mean())


def ink_ok(img: Image, cfg: GateConfig) -> bool:
    """Cheap pre-gate: reject a blank or a gross smear on ink density alone (no model needed)."""
    return cfg.ink_min <= ink_fraction(img) <= cfg.ink_max


def cer_ok(cer: float, cfg: GateConfig) -> bool:
    """Keep a line only if the recognizer read it back close enough to its intended text."""
    return cer <= cfg.cer_max
