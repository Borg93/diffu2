"""Rule-based wrong candidates for Jev's negatives, wrong by construction: each rule (:data:`PERTURBATIONS`) changes a
line's GT text as a reader of period hands errs, or gives None where it cannot (no digit to change, one word to drop).
Each draws from the random source it is given, so a line's negatives are fixed by its seed."""

from __future__ import annotations

import random
from collections.abc import Callable
from enum import StrEnum


class Perturbation(StrEnum):
    """How a candidate is made wrong."""

    CONFUSABLE = "confusable"  # a letter a reader confuses in period hands
    INSERT = "insert"
    DROP = "drop"
    WORD_DROP = "word-drop"
    WORD_SWAP = "word-swap"
    DIGIT = "digit"
    TRUNCATE = "truncate"


# Letters a reader confuses in period hands (n and u, e and c, long s and f) or reads without their marks
CONFUSABLES: dict[str, str] = {
    "n": "u",
    "u": "n",
    "e": "c",
    "c": "e",
    "ſ": "f",
    "f": "ſ",
    "ä": "a",
    "å": "a",
    "ö": "o",
}

type Perturb = Callable[[str, random.Random], str | None]


def _confusable(text: str, rng: random.Random) -> str | None:
    places = [k for k, letter in enumerate(text) if letter in CONFUSABLES]
    if not places:
        return None
    k = rng.choice(places)
    return text[:k] + CONFUSABLES[text[k]] + text[k + 1 :]


def _insert(text: str, rng: random.Random) -> str | None:
    letters = sorted({letter for letter in text if letter.isalpha()})
    if not letters:
        return None
    k = rng.randrange(len(text) + 1)
    return text[:k] + rng.choice(letters) + text[k:]


def _drop(text: str, rng: random.Random) -> str | None:
    places = [k for k, letter in enumerate(text) if not letter.isspace()]
    if len(places) < 2:
        return None
    k = rng.choice(places)
    return text[:k] + text[k + 1 :]


def _word_drop(text: str, rng: random.Random) -> str | None:
    words = text.split()
    if len(words) < 2:
        return None
    k = rng.randrange(len(words))
    return " ".join(words[:k] + words[k + 1 :])


def _word_swap(text: str, rng: random.Random) -> str | None:
    words = text.split()
    pairs = [k for k in range(len(words) - 1) if words[k] != words[k + 1]]
    if not pairs:
        return None
    k = rng.choice(pairs)
    words[k], words[k + 1] = words[k + 1], words[k]
    return " ".join(words)


def _digit(text: str, rng: random.Random) -> str | None:
    places = [k for k, letter in enumerate(text) if letter in "0123456789"]
    if not places:
        return None
    k = rng.choice(places)
    other = rng.choice([digit for digit in "0123456789" if digit != text[k]])
    return text[:k] + other + text[k + 1 :]


def _truncate(text: str, rng: random.Random) -> str | None:
    words = text.split()
    if len(words) < 2:
        return None
    return " ".join(words[1:] if rng.random() < 0.5 else words[:-1])


PERTURBATIONS: dict[Perturbation, Perturb] = {
    Perturbation.CONFUSABLE: _confusable,
    Perturbation.INSERT: _insert,
    Perturbation.DROP: _drop,
    Perturbation.WORD_DROP: _word_drop,
    Perturbation.WORD_SWAP: _word_swap,
    Perturbation.DIGIT: _digit,
    Perturbation.TRUNCATE: _truncate,
}


def perturbed(text: str, count: int, rng: random.Random) -> list[tuple[Perturbation, str]]:
    """Up to ``count`` wrong candidates for ``text``, each by another rule (drawn in ``rng``'s order), each differing
    from ``text`` and from the others."""
    kinds = list(Perturbation)
    rng.shuffle(kinds)
    made: list[tuple[Perturbation, str]] = []
    for kind in kinds:
        if len(made) == count:
            break
        candidate = PERTURBATIONS[kind](text, rng)
        if candidate is not None and candidate != text and candidate not in {wrong for _, wrong in made}:
            made.append((kind, candidate))
    return made
