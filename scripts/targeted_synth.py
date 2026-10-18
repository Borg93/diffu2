#!/usr/bin/env python
"""Turn an error report into a TARGETED text list for diffu-synth — the "sample up" half of error-driven aug.

Reads scripts/error_analysis.py's error_report.json and resamples the real text corpus so the synth we
generate over-represents exactly what the recognizer fails on. Each candidate line is scored by its total
"weakness mass" = sum over its characters of that character's measured error rate; sampling weight ∝ score.

Because the score SUMS (not averages) per-char error, it targets two axes at once:
  * content — lines rich in weak characters (digits, capitals, "( )", "'", ";") score high;
  * length  — longer lines accumulate more mass, so the weak long-line bucket is over-sampled too.

    uv run python scripts/targeted_synth.py --report runs/htr_ab/error_report.json --n 20000
      -> writes runs/htr_ab/targeted_texts.txt ; then:
    diffu-synth --ckpt <base> --lora <adapter> --hf-dataset Teklia/IAM-line --texts runs/htr_ab/targeted_texts.txt --out ...
"""

from __future__ import annotations

import argparse
import json
import random


def char_class(c: str) -> str:
    if c == " ":
        return "space"
    if c.isdigit():
        return "digit"
    if c.isupper():
        return "upper"
    if c.islower():
        return "lower"
    return "punct"


def main() -> None:
    ap = argparse.ArgumentParser(description="Weakness-weighted text resampling for targeted synth.")
    ap.add_argument("--report", required=True, help="error_report.json from scripts/error_analysis.py")
    ap.add_argument("--hf-dataset", default="Teklia/IAM-line")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20000, help="how many targeted lines to draw")
    ap.add_argument("--boost", type=float, default=25.0, help="how hard to tilt toward weak content (0 = uniform)")
    ap.add_argument("--out", default="runs/htr_ab/targeted_texts.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    rep = json.load(open(args.report))
    class_err: dict[str, float] = rep["class_error_rate"]
    char_err: dict[str, float] = {c["char"]: c["err_rate"] for c in rep["weak_chars"]}
    baseline = class_err.get("lower", 0.09)  # "ordinary" difficulty — lowercase text the model already handles

    def excess(ch: str) -> float:
        # only content HARDER than ordinary lowercase drives sampling, so the sea of easy letters can't dominate
        return max(0.0, char_err.get(ch, class_err.get(char_class(ch), 0.05)) - baseline)

    ds = load_dataset(args.hf_dataset, split=args.split)
    col = "text" if "text" in ds.column_names else ds.column_names[-1]
    texts = [str(t).strip() for t in ds[col] if str(t).strip()]
    # score = floor (keeps diversity) + boost * summed excess-difficulty (targets weak chars AND, via the sum, long lines)
    scores = [1.0 + args.boost * sum(excess(c) for c in t) for t in texts]

    rng = random.Random(args.seed)
    chosen = rng.choices(texts, weights=scores, k=args.n)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(chosen) + "\n")

    # show the targeting EFFECT: weak-token fraction, uniform corpus vs targeted draw
    def frac(ts: list[str], pred) -> float:
        tot = sum(len(t) for t in ts) or 1
        return sum(sum(1 for c in t if pred(c)) for t in ts) / tot

    def meanlen(ts: list[str]) -> float:
        return sum(len(t) for t in ts) / (len(ts) or 1)

    print(f"wrote {len(chosen)} targeted lines -> {args.out}")
    print(f"{'':10} {'uniform':>9} {'targeted':>9}")
    print(f"{'digit %':10} {100*frac(texts, str.isdigit):9.2f} {100*frac(chosen, str.isdigit):9.2f}")
    print(f"{'upper %':10} {100*frac(texts, str.isupper):9.2f} {100*frac(chosen, str.isupper):9.2f}")
    print(f"{'punct %':10} {100*frac(texts, lambda c: char_class(c) == 'punct'):9.2f} {100*frac(chosen, lambda c: char_class(c) == 'punct'):9.2f}")
    print(f"{'mean len':10} {meanlen(texts):9.1f} {meanlen(chosen):9.1f}")


if __name__ == "__main__":
    main()
