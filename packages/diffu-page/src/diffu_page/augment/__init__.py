"""Scan-augmentation pipeline — turn a clean, seated page into one coherent *real-looking scan*.

Everything that makes a pristine composite look like an archive photograph lives here: photometric aging
(lighting, noise, foxing, bleed-through), the binding (spine shadow + localized warp + gutter focus), the
scanner bed (dark borders, frayed edges, page-holder clasps), and a final JPEG recompress. The whole
pipeline is driven by one :class:`AugmentConfig` value object — no magic numbers scattered around.

Public surface is intentionally small: :class:`AugmentConfig` and :func:`scan_pass`. ``compose`` seats the
lines on paper and hands the page here; the photometric stages leave labels untouched while the geometric
ones carry the polygons with them, so the ground truth stays valid throughout.

Split into cohesive submodules by concern — ``config``, ``photometric``, ``geometry``, ``binding``,
``bed``, ``marks`` and ``pipeline`` — this package re-exports the old flat-module surface.
"""

from __future__ import annotations

from .bed import PaperBounds
from .binding import center_shadow
from .config import AugmentConfig
from .pipeline import scan_pass

__all__ = ["AugmentConfig", "PaperBounds", "center_shadow", "scan_pass"]
