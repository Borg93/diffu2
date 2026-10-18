"""Corpus-proportion layout sampler — draw page specs in the GT-Lejonet type proportions.

The 59-volume corpus study measured the layout-type mix: prose-dominant, a sparse letter/blank tail, and a
structured register/ledger/grid minority. :class:`LayoutMix` encodes those measured weights and
:func:`sample_page_spec` draws a :class:`~diffu_page.layout.PageSpec` per page, so a generation run produces
a corpus-faithful dataset instead of a hand-set plan. Deterministic given the caller's ``random.Random``.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, Field

from .layout import PageSpec

# Layout kinds and their share of the corpus, measured from the 59-volume study (prose ~54%, sparse
# letter/blank tail ~24%, structured register/ledger/grid ~22%). Tune to reweight a generation run.
_CORPUS_WEIGHTS: dict[str, float] = {
    "prose_medium": 0.33,
    "prose_dense": 0.20,
    "letter": 0.12,
    "blank": 0.11,
    "register": 0.07,
    "landskap": 0.06,  # sparse display-headed tax/cattle roll (boskapslängd/landskapshandling)
    "ledger": 0.06,
    "semi_table": 0.05,
}

# Kinds that may be emitted as a two-page spread. The ``spread`` template flows plain paragraphs into
# each leaf (templates._spread -> _spread_leaf -> _flow_paragraphs), so only the prose-like kinds fit;
# the tabular kinds (register / landskap / ledger / semi_table) lay out rules + typed cells that
# ``_spread`` cannot reproduce per leaf, so they stay single-leaf until it supports them.
SPREAD_KINDS: frozenset[str] = frozenset({"prose_medium", "prose_dense", "letter", "blank"})

_SPREAD_GUTTER_FRAC = 0.04  # extra canvas for the central gutter, as a fraction of the single-leaf width


class LayoutMix(BaseModel):
    """Per-kind sampling weights (need not sum to 1 — normalized at draw time). Defaults = the corpus mix."""

    model_config = {"frozen": True}

    weights: dict[str, float] = Field(default_factory=lambda: dict(_CORPUS_WEIGHTS))
    # Fraction of the prose-like kinds (SPREAD_KINDS) emitted as a two-page opening — most real
    # bound-volume scans are spreads. 0 disables spreads; 1 spreads every eligible page.
    spread_frac: float = Field(default=0.35, ge=0.0, le=1.0)


DEFAULT_MIX = LayoutMix()


def sample_kind(rng: random.Random, mix: LayoutMix | None = None) -> str:
    """Draw one layout kind weighted by the (corpus-proportion) mix."""
    weights = (mix or DEFAULT_MIX).weights
    kinds = list(weights)
    return rng.choices(kinds, weights=[weights[k] for k in kinds], k=1)[0]


def _maybe_catchword(rng: random.Random) -> str | None:
    return rng.choice(["Anno", "och", "den", "att", "som"]) if rng.random() < 0.3 else None


def _spec_for_kind(kind: str, rng: random.Random) -> PageSpec:
    """A ``PageSpec`` for one sampled kind, with corpus-typical density / template / furniture."""
    base = PageSpec(  # shared page furniture; each kind overrides its template/density via model_copy
        width=1654,
        height=2339,
        recto=rng.random() < 0.5,
        page_number=str(rng.randint(1, 400)) if rng.random() < 0.7 else None,
    )
    if kind == "letter":
        return base.model_copy(
            update={
                "template": "letter",
                "line_height": rng.randint(100, 124),  # GT scale: big script, ~13 lines/page (not ~40)
                "catchword": _maybe_catchword(rng),
            }
        )
    if kind == "blank":
        return base.model_copy(update={"template": "blank_leaf", "line_height": 96})
    if kind == "register":
        return base.model_copy(
            update={"template": "register", "line_height": rng.randint(62, 76), "line_gap": 0.5}
        )
    if kind == "landskap":  # sparse display-headed tax/cattle roll
        return base.model_copy(
            update={
                "template": "landskap",
                "line_height": rng.randint(60, 74),
                "line_gap": rng.uniform(0.5, 0.7),
            }
        )
    if kind == "ledger":
        return base.model_copy(
            update={
                "template": "ledger",
                "line_height": rng.randint(54, 66),
                "line_gap": 0.35,
                "table_cols": rng.randint(4, 6),
            }
        )
    if kind == "semi_table":
        return base.model_copy(
            update={
                "template": "semi_table",
                "line_height": rng.randint(52, 64),
                "table_rows": rng.randint(14, 20),
                "table_cols": rng.randint(5, 7),
            }
        )
    dense = kind == "prose_dense"  # prose (medium or dense) with sprinkled text furniture
    return base.model_copy(
        update={
            "template": "single_column",
            # GT scale: real pages fill the column with big script, ~13 lines (dense ~20), not tiny 40+ lines
            "line_height": rng.randint(80, 94) if dense else rng.randint(98, 122),
            "line_gap": 0.2 if dense else 0.4,
            "catchword": _maybe_catchword(rng),
            "underline_prob": 0.2 if rng.random() < 0.3 else 0.0,
            "section_headings": ["Resolutio", "Utslag:"] if rng.random() < 0.25 else [],
            "enum_prob": 0.5 if rng.random() < 0.2 else 0.0,
            "binding": rng.random() < 0.2,
        }
    )


def _as_spread(spec: PageSpec) -> PageSpec:
    """Rebuild a single-leaf spec as a two-page opening: the ``spread`` template on a canvas widened to
    two leaves plus a central gutter. The kind's density fields (line_height / line_gap / ...) carry
    over — ``_spread`` flows each leaf with them. ``binding`` is forced off (a spread's spine is the
    CENTRAL gutter the template keeps text-free, not a single-edge inner gutter) and ``page_number`` is
    cleared because ``_spread`` writes its own verso/recto pair into the outer corners."""
    return spec.model_copy(
        update={
            "template": "spread",
            "width": 2 * spec.width + int(_SPREAD_GUTTER_FRAC * spec.width),
            "binding": False,
            "page_number": None,
        }
    )


def sample_page_spec(rng: random.Random, mix: LayoutMix | None = None) -> tuple[str, PageSpec]:
    """Draw a ``(kind, PageSpec)`` pair in corpus proportions — the generation entry point.

    Prose-like kinds (:data:`SPREAD_KINDS`) come out as a two-page spread with probability
    ``mix.spread_frac``; the tabular kinds always stay single-leaf (see :data:`SPREAD_KINDS`).
    """
    kind = sample_kind(rng, mix)
    spec = _spec_for_kind(kind, rng)
    if kind in SPREAD_KINDS and rng.random() < (mix or DEFAULT_MIX).spread_frac:
        spec = _as_spread(spec)
    return kind, spec
