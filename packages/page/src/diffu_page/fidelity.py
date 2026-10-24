"""How closely drawn hands follow: a drawn crop's hand measured beside its real crop's and its reference's
(``diffu_page.captions.hand``), and over a run, the medians of those distances and how much of the references'
variation the drawings carry over. diffu-evaluate measures every drawing with it, and ``diffu_page.cli.hands``
measures a finished run's saved drawings."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import linear_regression, median, pstdev

from PIL import Image
from pydantic import BaseModel

from diffu_page.captions.hand import Box, HandDistance, HandLook, hand_distance, hand_look, lab
from diffu_page.captions.prompt import LineRecord
from diffu_page.crops.grid import as_reference

MIN_SPREAD = 1.0  # a carry-over needs references that vary by this much (L* or b* units, degrees of slant)


class Hands(BaseModel):
    """A drawn crop's hand beside its real crop's, both measured in the crop's line boxes, and beside its
    reference's, measured in the reference's at the reference size; and how far apart they are."""

    drawn: HandLook
    real: HandLook
    reference: HandLook | None = None
    to_real: HandDistance
    to_reference: HandDistance | None = None
    real_to_reference: HandDistance | None = None  # the pair's own gap: how alike the two real hands are


def crop_hands(
    drawn: Image.Image,
    real: Image.Image,
    lines: Sequence[LineRecord],
    *,
    reference: Image.Image | None = None,
    reference_lines: Sequence[Box] | None = None,
) -> Hands:
    """The hands of a drawn crop, its real crop and, given its line boxes, its reference."""
    boxes = [line.bbox for line in lines]
    drawn_hand, real_hand = hand_look(drawn, boxes), hand_look(real, boxes)
    if reference is None or not reference_lines:
        return Hands(drawn=drawn_hand, real=real_hand, to_real=hand_distance(drawn_hand, real_hand))
    reference_hand = hand_look(as_reference(reference), reference_lines)
    return Hands(
        drawn=drawn_hand,
        real=real_hand,
        reference=reference_hand,
        to_real=hand_distance(drawn_hand, real_hand),
        to_reference=hand_distance(drawn_hand, reference_hand),
        real_to_reference=hand_distance(real_hand, reference_hand),
    )


class CarryOver(BaseModel):
    """How much of the references' variation reaches the hands across the crops: the least-squares slope of a hand's
    measure on its reference's (1: follows the reference fully; 0: ignores it). Colours by CIE L*a*b* lightness L* and
    yellowness b* (cream and brown against grey and black). None where fewer than three crops have the measure or the
    references vary by less than ``MIN_SPREAD`` (the yellowness of grey inks is 0 plus rounding).

    Unlike a median distance, a slope is not fooled by a model's default: ink drawn near-black whatever the reference
    lands close to every black-inked reference, but its slope stays at 0."""

    paper_lightness: float | None
    paper_yellowness: float | None
    ink_lightness: float | None
    ink_yellowness: float | None
    slant: float | None


def _slope(pairs: Sequence[tuple[float, float]]) -> float | None:
    references = [x for x, _ in pairs]
    if len(pairs) < 3 or pstdev(references) < MIN_SPREAD:
        return None
    return linear_regression(references, [y for _, y in pairs]).slope


def carry_over(pairs: Sequence[tuple[HandLook, HandLook]]) -> CarryOver:
    """How much of the references' variation reaches the hands in ``pairs`` of (reference, hand)."""

    def colour(field: str, axis: int) -> float | None:
        measured = [(getattr(reference, field), getattr(hand, field)) for reference, hand in pairs]
        return _slope([(lab(a)[axis], lab(b)[axis]) for a, b in measured if a is not None and b is not None])

    slants = [
        (reference.slant, hand.slant)
        for reference, hand in pairs
        if reference.slant is not None and hand.slant is not None
    ]
    return CarryOver(
        paper_lightness=colour("paper", 0),
        paper_yellowness=colour("paper", 2),
        ink_lightness=colour("ink", 0),
        ink_yellowness=colour("ink", 2),
        slant=_slope(slants),
    )


class HandFidelity(BaseModel):
    """Medians over the drawn crops of each ``HandDistance`` measure: a drawn hand's distance to its real crop's;
    over the crops with a measured reference, its distance to its own reference's, to the other crops' references
    (chance: no lower than this when the reference's hand is ignored), and the real crop's distance to its
    reference's. And the carry-over from the references to the drawn hands, beside the carry-over to the real crops
    (what the references can tell of the real hands: the ceiling)."""

    to_real: HandDistance
    to_reference: HandDistance | None = None
    to_other_references: HandDistance | None = None
    real_to_reference: HandDistance | None = None
    carry_over: CarryOver | None = None
    real_carry_over: CarryOver | None = None


def _median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return median(present) if present else None


def _medians(distances: list[HandDistance]) -> HandDistance:
    return HandDistance.model_validate(
        {field: _median([getattr(distance, field) for distance in distances]) for field in HandDistance.model_fields}
    )


def hand_fidelity(hands: Sequence[Hands]) -> HandFidelity:
    """The medians of ``hands`` (at least one)."""
    fidelity = HandFidelity(to_real=_medians([hand.to_real for hand in hands]))
    referenced = [hand for hand in hands if hand.reference is not None]
    if not referenced:
        return fidelity
    others = [
        hand_distance(hand.drawn, other.reference)
        for hand in referenced
        for other in referenced
        if other is not hand and other.reference is not None
    ]
    return fidelity.model_copy(
        update={
            "to_reference": _medians([hand.to_reference for hand in referenced if hand.to_reference is not None]),
            "to_other_references": _medians(others) if others else None,
            "real_to_reference": _medians(
                [hand.real_to_reference for hand in referenced if hand.real_to_reference is not None]
            ),
            "carry_over": carry_over([(hand.reference, hand.drawn) for hand in referenced if hand.reference]),
            "real_carry_over": carry_over([(hand.reference, hand.real) for hand in referenced if hand.reference]),
        }
    )
