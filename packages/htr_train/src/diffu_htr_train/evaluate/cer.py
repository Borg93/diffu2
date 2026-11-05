"""Error rates as each side scores itself: the Swedish Lion TrOCR's (``jiwer.cer`` a line, whitespace normalised, the
mean over the lines) and OpenBMB's (``docs/swift_minicpm_ocr_french_en.md``: NFC, the ends trimmed, inner spaces kept;
the corpus CER and WER, edits over reference characters or words); and the baseline's training normalisation
(:func:`trocr_normalize`) to score both sides with too."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

from pydantic import BaseModel


def edits(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """The Levenshtein distance between two sequences (a string's characters, a text's words): substitutions,
    insertions and deletions."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def whitespace(text: str) -> str:
    """``text`` with every run of whitespace one space, none at the ends (the baseline's evaluation)."""
    return re.sub(r"\s+", " ", text).strip()


def nfc_trimmed(text: str) -> str:
    """``text`` as OpenBMB scores it: NFC, its ends trimmed, case, accents, punctuation and inner spaces kept."""
    return unicodedata.normalize("NFC", text).strip()


_FRACTIONS = [
    ("½", "1/2"), ("↉", "0/3"), ("⅓", "1/3"), ("⅔", "2/3"), ("¼", "1/4"), ("¾", "3/4"), ("⅕", "1/5"), ("⅖", "2/5"),
    ("⅗", "3/5"), ("⅘", "4/5"), ("⅙", "1/6"), ("⅚", "5/6"), ("⅐", "1/7"), ("⅛", "1/8"), ("⅜", "3/8"), ("⅝", "5/8"),
    ("⅞", "7/8"), ("⅑", "1/9"), ("⅒", "1/10"),
]  # fmt: skip


def trocr_normalize(text: str) -> str:
    """``text`` as the Swedish Lion TrOCR baseline normalises its training text (its ``train.py`` ``normalize``):
    fractions spelled out, ``¬`` joined, ``„`` and ``~`` dropped, quotes and middle dots made plain, runs of dots one
    ellipsis, dashes between spaces and runs of dashes one em dash, whitespace collapsed, NFKC."""
    for fraction, spelled in _FRACTIONS:
        text = text.replace(fraction, f" {spelled}")
    text = re.sub(r"\s*¬", "¬", text)
    text = re.sub(r"¬+", r"¬", text)
    text = text.replace("„", "")
    text = text.replace("”", '"').replace("“", '"').replace("´", "'")
    text = text.replace("·", ".").replace("‧", ".")
    text = text.replace("~", "")
    text = re.sub(r"(\. ?)(\. ?)(\. ?)+", "... ", text)
    text = text.replace("…", "... ")
    text = text.replace("﹘", " — ")
    text = text.replace("_,", "")
    text = re.sub(r"(^| )(-|_|—|‒|―|–)( |$)", " — ", text)
    text = re.sub(r"((-|_|—|‒|―|–) ?)((-|_|—|‒|―|–) ?)+", " — ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip("— ")
    return unicodedata.normalize("NFKC", text)


class Scored(BaseModel):
    """What a reader's lines came to: their number, the mean of each line's CER (an empty reference line counts 0
    when read empty, else 1), the corpus CER (all edits over all reference characters) and the corpus WER (word edits,
    the words split at whitespace, over all reference words)."""

    lines: int
    mean_cer: float
    corpus_cer: float
    corpus_wer: float


def score(pairs: Iterable[tuple[str, str]], normalize: Callable[[str], str] = whitespace) -> Scored:
    """Each (reference, prediction) pair scored after ``normalize`` on both sides."""
    rates: list[float] = []
    total_edits = total_chars = word_edits = total_words = 0
    for reference, prediction in pairs:
        ref, hyp = normalize(reference), normalize(prediction)
        distance = edits(ref, hyp)
        rates.append(distance / len(ref) if ref else float(bool(hyp)))
        total_edits += distance
        total_chars += len(ref)
        word_edits += edits(ref.split(), hyp.split())
        total_words += len(ref.split())
    return Scored(
        lines=len(rates),
        mean_cer=sum(rates) / len(rates) if rates else 0.0,
        corpus_cer=total_edits / total_chars if total_chars else 0.0,
        corpus_wer=word_edits / total_words if total_words else 0.0,
    )


def by_subset(rows: Iterable[tuple[str, str, str]], normalize: Callable[[str], str] = whitespace) -> dict[str, Scored]:
    """Each subset's (subset, reference, prediction) rows scored apart, and all of them as ``"all"``."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for subset, reference, prediction in rows:
        groups[subset].append((reference, prediction))
        groups["all"].append((reference, prediction))
    return {subset: score(pairs, normalize) for subset, pairs in sorted(groups.items())}
