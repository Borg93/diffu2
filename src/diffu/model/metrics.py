"""Lightweight metrics: character edit distance and CER."""

from __future__ import annotations

from collections.abc import Sequence


def levenshtein(a: str, b: str) -> int:
    """Character edit distance between ``a`` and ``b``."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds: Sequence[str], gts: Sequence[str]) -> float:
    """Micro-averaged CER over a batch: total edits / total ground-truth characters (an empty ground truth
    counts as one character). A macro mean of per-line ratios would over-weight short lines."""
    total_d, total_n = 0, 0
    for p, g in zip(preds, gts, strict=True):
        total_d += levenshtein(p, g)
        total_n += max(len(g), 1)
    return total_d / max(total_n, 1)


def char_error_rate(pred: str, gt: str) -> float:
    """One line's CER: Levenshtein(pred, gt) / len(gt). 0 = perfect, ~1+ = illegible; an empty ground truth
    scores 0 for an empty prediction, else 1."""
    if not gt:
        return 1.0 if pred else 0.0
    return levenshtein(pred, gt) / len(gt)
