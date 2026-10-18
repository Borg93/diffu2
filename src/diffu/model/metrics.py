"""Lightweight metrics. CER = mean Levenshtein(pred, gt) / len(gt) over a batch."""

from __future__ import annotations

from collections.abc import Sequence


def _levenshtein(a: str, b: str) -> int:
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
    total_d, total_n = 0, 0
    for p, g in zip(preds, gts, strict=False):
        total_d += _levenshtein(p, g)
        total_n += max(len(g), 1)
    return total_d / max(total_n, 1)
