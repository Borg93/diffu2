#!/usr/bin/env python
"""Materialize the FULL IAM-line train split to disk for the scale-up experiment (all 6482 real lines).

We had been training on a 2000-line subset (runs/htr_ab/real). For "take all IAM as much as possible" the
recognizer arms need the whole train split on disk (the TrOCR dataset reads row["image"] as a file path).
Reuses the existing held-out test (runs/htr_ab/test) unchanged for comparability.

    uv run python scripts/prepare_iam_full.py   # -> runs/htr_ab/real_all/{lines,train.jsonl}
"""

from __future__ import annotations

import json
import os

from datasets import load_dataset

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Materialize an IAM-line split (images + jsonl) to disk.")
    ap.add_argument("--split", default="train", help="IAM split: train / validation / test")
    ap.add_argument("--out", default="runs/htr_ab/real_all")
    ap.add_argument("--name", default="train.jsonl", help="manifest filename")
    args = ap.parse_args()

    ds = load_dataset("Teklia/IAM-line", split=args.split)
    lines_dir = os.path.join(args.out, "lines")
    os.makedirs(lines_dir, exist_ok=True)
    path = os.path.join(args.out, args.name)
    with open(path, "w", encoding="utf-8") as mf:
        for i, row in enumerate(ds):
            text = str(row["text"]).strip()
            if not text:
                continue
            dst = os.path.join(lines_dir, f"{i:07d}.png")
            row["image"].convert("RGB").save(dst)
            mf.write(json.dumps({"image": os.path.abspath(dst), "text": text}, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                print(f"  {i}/{len(ds)}", flush=True)
    print(f"wrote IAM {args.split} -> {path} ({len(ds)} lines)", flush=True)


if __name__ == "__main__":
    main()
