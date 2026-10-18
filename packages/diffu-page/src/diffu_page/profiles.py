"""Named scan profiles — presets that span the real GT-Lejonet tone/contrast range.

Real archival scans do not sit at one tone: the corpus runs from a high-contrast **bitonal** microfilm,
through a **warm sepia** colour flatbed and a low-contrast **faded grey**, to a strongly **yellowed** leaf.
A synthetic set that only ever renders one tone teaches an HTR model a narrow paper prior; drawing a random
profile per page widens it.

A :class:`ScanProfile` bundles the two controls a tone needs: a :class:`~diffu_page.paper.PaperTone` (paper
hue + luminance) and an :class:`~diffu_page.augment.AugmentConfig` (the aging pipeline). Pass both to
``compose_page(..., paper_tone=profile.paper_tone, aug=profile.aug)``.
"""

from __future__ import annotations

import random

from pydantic import BaseModel

from .augment import AugmentConfig
from .paper import PaperTone


class ScanProfile(BaseModel):
    """A named paper-tone + aging preset. ``weight`` biases how often :func:`sample_profile` draws it."""

    model_config = {"frozen": True}

    name: str
    paper_tone: PaperTone
    aug: AugmentConfig
    weight: float = 1.0


# Physical-handling + document-mark rates shared across profiles, so every tone shows the same repertoire of
# gloved fingers / holes / curled corners / folds / stamps regardless of colour. Each profile then retunes
# only the tone-specific knobs (foxing load, wash, bleed, microfilm, ink fade) via ``model_copy``.
_MARKS = AugmentConfig(
    glove_prob=0.22,
    hole_prob=0.10,
    curl_prob=0.20,
    fold_prob=0.28,
    stamp_prob=0.14,
    paraph_prob=0.18,
    margin_mark_prob=0.15,
)

SCAN_PROFILES: list[ScanProfile] = [
    ScanProfile(
        name="bitonal",
        paper_tone=PaperTone(base_tone=(226, 247), tint=(0.98, 1.0, 1.0)),  # bright near-neutral sheet
        # filmed -> high-contrast grey with grain + tramlines + frame; almost no bleed/wash
        aug=_MARKS.model_copy(
            update={"microfilm_prob": 1.0, "bleed": (0.0, 0.05), "wash": (0.0, 0.10), "ink_fade": (0.0, 0.08)}
        ),
        weight=0.9,
    ),
    ScanProfile(
        name="warm_sepia",
        paper_tone=PaperTone(base_tone=(204, 231), tint=(0.78, 0.90, 1.02)),  # warm brown flatbed
        aug=_MARKS.model_copy(
            update={
                "microfilm_prob": 0.0,
                "bleed": (0.03, 0.14),
                "wash": (0.0, 0.30),
                "ink_fade": (0.0, 0.16),
            }
        ),
    ),
    ScanProfile(
        name="faded_gray",
        paper_tone=PaperTone(base_tone=(150, 186), tint=(0.98, 0.99, 0.98)),  # dim, low-contrast grey stock
        # heavy damp wash pulls toward mid-grey; ink faded toward the paper
        aug=_MARKS.model_copy(
            update={
                "microfilm_prob": 0.0,
                "bleed": (0.02, 0.10),
                "wash": (0.28, 0.52),
                "ink_fade": (0.14, 0.34),
            }
        ),
    ),
    ScanProfile(
        name="yellowed",
        paper_tone=PaperTone(base_tone=(198, 225), tint=(0.62, 0.95, 1.06)),  # strongly yellowed leaf
        aug=_MARKS.model_copy(
            update={
                "microfilm_prob": 0.0,
                "bleed": (0.03, 0.14),
                "wash": (0.0, 0.24),
                "ink_fade": (0.0, 0.14),
            }
        ),
    ),
]

PROFILES_BY_NAME: dict[str, ScanProfile] = {p.name: p for p in SCAN_PROFILES}
PROFILE_NAMES: list[str] = [p.name for p in SCAN_PROFILES]


def profile(name: str) -> ScanProfile:
    """Look up a scan profile by name (``bitonal`` | ``warm_sepia`` | ``faded_gray`` | ``yellowed``)."""
    if name not in PROFILES_BY_NAME:
        raise ValueError(f"unknown scan profile {name!r}; options: {PROFILE_NAMES}")
    return PROFILES_BY_NAME[name]


def sample_profile(rng: random.Random) -> ScanProfile:
    """Draw a scan profile at random, biased by each profile's ``weight``."""
    return rng.choices(SCAN_PROFILES, weights=[p.weight for p in SCAN_PROFILES], k=1)[0]
