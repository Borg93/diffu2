#!/usr/bin/env python
"""Sample IAM texts for a fair diffu-vs-DiffusionPen comparison — the SAME content rendered by both models.

Writes two files: lines.txt (full IAM sentences, for the line-level downstream-HTR protocol) and words.txt
(single IAM words, for the word-level image-quality protocol). Both models render these exact texts so the
only difference is the generator, not the content.

    uv run python scripts/sample_iam_texts.py --n-lines 20000 --n-words 5000 --out runs/htr_ab/compare
"""

from __future__ import annotations

import argparse
import os
import random


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample matched IAM line/word texts for model comparison.")
    ap.add_argument("--hf-dataset", default="Teklia/IAM-line")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-lines", type=int, default=20000)
    ap.add_argument("--n-words", type=int, default=5000)
    ap.add_argument("--out", default="runs/htr_ab/compare")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset, split=args.split)
    col = "text" if "text" in ds.column_names else ds.column_names[-1]
    lines = [str(t).strip() for t in ds[col] if str(t).strip()]
    rng = random.Random(args.seed)

    os.makedirs(args.out, exist_ok=True)
    # lines: sample with replacement up to n-lines (corpus is ~6482 unique -> repeats, same as our synth)
    chosen_lines = rng.choices(lines, k=args.n_lines)
    with open(os.path.join(args.out, "lines.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(chosen_lines) + "\n")

    # words: unique-ish tokens (drop pure punctuation), sampled
    words = [w for ln in lines for w in ln.split() if any(c.isalnum() for c in w)]
    chosen_words = rng.choices(words, k=args.n_words)
    with open(os.path.join(args.out, "words.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(chosen_words) + "\n")

    print(f"wrote {len(chosen_lines)} lines -> {args.out}/lines.txt  (from {len(lines)} unique)")
    print(f"wrote {len(chosen_words)} words -> {args.out}/words.txt  (from {len(set(words))} unique)")


if __name__ == "__main__":
    main()
