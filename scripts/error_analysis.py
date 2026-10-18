#!/usr/bin/env python
"""Where does an HTR recognizer fail? — the analysis half of error-driven ("hard-example") augmentation.

Runs a recognizer over a test manifest, then breaks the error down along the axes that tell you WHAT to
synthesize more of: line length, character class (digit/upper/punct/...), and per-character error rate.
Emits a human report AND a machine-readable JSON of the weak spots, which scripts/targeted_synth.py reads to
bias generation toward the model's weaknesses.

    uv run python scripts/error_analysis.py --htr runs/htr_ab/htr_real --manifest runs/htr_ab/test/test.jsonl
      -> prints report, writes runs/htr_ab/error_report.json
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter

from diffu.eval import HTRRecognizer, _open
from diffu.model.metrics import cer as _cer


def line_cer(gt: str, pred: str) -> float:
    return _cer([pred], [gt])


def char_class(c: str) -> str:
    if c == " ":
        return "space"
    if c.isdigit():
        return "digit"
    if c.isupper():
        return "upper"
    if c.islower():
        return "lower"
    return "punct"  # everything else: punctuation / symbols


def align_errors(gt: str, pred: str) -> list[tuple[str, bool]]:
    """Levenshtein-align gt->pred; return (gt_char, errored) per GT char (sub/del count as errored)."""
    n, m = len(gt), len(pred)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    # backtrace
    out: list[tuple[str, bool]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if gt[i - 1] == pred[j - 1] else 1):
            out.append((gt[i - 1], gt[i - 1] != pred[j - 1]))  # match or substitution
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((gt[i - 1], True))  # deletion of a GT char
            i -= 1
        else:
            j -= 1  # insertion — no GT char to attribute
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Break an HTR recognizer's error down by length / char-class / char.")
    ap.add_argument("--htr", required=True, help="trained recognizer dir")
    ap.add_argument("--manifest", required=True, help="test.jsonl {image,text}")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="runs/htr_ab/error_report.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]
    rec = HTRRecognizer(model_id=args.htr, batch_size=16)
    preds = rec([_open(r["image"]) for r in rows])
    gts = [r["text"] for r in rows]

    lines = [(g, p, line_cer(g, p)) for g, p in zip(gts, preds)]
    overall = _cer(preds, gts)

    # 1) CER by line length (chars)
    len_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 9999)]
    by_len: dict[str, list[float]] = {f"{a}-{b}": [] for a, b in len_bins}
    for g, _p, c in lines:
        for a, b in len_bins:
            if a <= len(g) < b:
                by_len[f"{a}-{b}"].append(c)
                break

    # 2) CER for lines that CONTAIN each char class (vs the overall) — which content is hard
    has_class: dict[str, list[float]] = {k: [] for k in ("digit", "upper", "punct")}
    for g, _p, c in lines:
        classes = {char_class(ch) for ch in g}
        for k in has_class:
            if k in classes:
                has_class[k].append(c)

    # 3) per-character error rate (which GT chars get mangled most), aggregated by class too
    occ: Counter[str] = Counter()
    err: Counter[str] = Counter()
    cls_occ: Counter[str] = Counter()
    cls_err: Counter[str] = Counter()
    for g, p, _c in lines:
        for ch, bad in align_errors(g, p):
            occ[ch] += 1
            cls_occ[char_class(ch)] += 1
            if bad:
                err[ch] += 1
                cls_err[char_class(ch)] += 1

    def rate(e: Counter[str], o: Counter[str], key: str) -> float:
        return e[key] / o[key] if o[key] else 0.0

    # weak chars: seen >=20 times, ranked by error rate
    weak_chars = sorted(
        [(ch, err[ch] / occ[ch], occ[ch]) for ch in occ if occ[ch] >= 20],
        key=lambda t: -t[1],
    )[:15]
    worst_lines = sorted(lines, key=lambda t: -t[2])[:8]

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print(f"=== ERROR ANALYSIS: {args.htr}  (overall CER {100*overall:.2f}%, n={len(rows)}) ===")
    print("\n-- CER by line length --")
    for k, v in by_len.items():
        print(f"  len {k:>7}  n={len(v):4d}  CER {100*avg(v):.2f}%")
    print("\n-- CER of lines CONTAINING a class (vs overall) --")
    for k, v in has_class.items():
        print(f"  has {k:6}  n={len(v):4d}  CER {100*avg(v):.2f}%")
    print("\n-- per-CLASS char error rate --")
    for k in ("digit", "upper", "punct", "lower", "space"):
        print(f"  {k:6}  seen={cls_occ[k]:6d}  err-rate {100*rate(cls_err, cls_occ, k):.2f}%")
    print("\n-- weakest characters (seen>=20) --")
    for ch, r, o in weak_chars:
        name = unicodedata.name(ch, "?")
        print(f"  {ch!r:5}  err {100*r:.1f}%  (n={o})  {name}")
    print("\n-- worst lines --")
    for g, p, c in worst_lines:
        print(f"  CER {100*c:.0f}%  gt={g!r}\n            pred={p!r}")

    report = {
        "htr": args.htr,
        "overall_cer": overall,
        "cer_by_length": {k: avg(v) for k, v in by_len.items()},
        "cer_has_class": {k: avg(v) for k, v in has_class.items()},
        "class_error_rate": {k: rate(cls_err, cls_occ, k) for k in ("digit", "upper", "punct", "lower", "space")},
        "weak_chars": [{"char": ch, "err_rate": r, "n": o} for ch, r, o in weak_chars],
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
