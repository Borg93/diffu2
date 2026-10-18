#!/usr/bin/env python
"""Photometrically harmonize diffu-synth line images to the real-IAM scan manifold (CPU, no regen).

The fidelity probe showed the synth is *label-clean* (a strong recognizer reads it at 2.90% CER) but sits
off-manifold on the IMAGE side: ~2x denser ink and ~1.9x contrast vs real IAM scans. That covariate shift is
the leading suspect for "more synth hurts a data-rich recognizer." This script pulls the one lever nobody
tried: per-image standardize each synth PNG's grayscale luminance to the real-IAM (mean, std), which lightens
the ink and compresses contrast toward real — no re-generation, purely a transform of the existing pixels.

    uv run python scripts/harmonize_synth.py \
        --synth runs/htr_ab/synth_fixed/train.jsonl \
        --real  runs/htr_ab/real/train.jsonl \
        --out   runs/htr_ab/synth_harmonized

Writes <out>/images/*.png + <out>/train.jsonl (same labels), and prints before/after ink/contrast so you can
confirm the synth moved toward real.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image


def _gray_stats(path: str) -> tuple[float, float, float]:
    """(mean_luminance, std, ink_fraction<128) of an image in grayscale."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"), dtype=np.float32)
    return float(g.mean()), float(g.std()), float((g < 128).mean())


def _sample_target(manifest: str, n: int = 300) -> tuple[float, float]:
    """Mean grayscale (mean, std) over the first ``n`` real lines — the harmonization target."""
    rows = [json.loads(line) for line in open(manifest)][:n]
    means, stds = zip(*[(_gray_stats(r["image"])[0], _gray_stats(r["image"])[1]) for r in rows])
    return float(np.mean(means)), float(np.mean(stds))


def harmonize(synth_manifest: str, real_manifest: str, out_dir: str) -> None:
    tgt_mean, tgt_std = _sample_target(real_manifest)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = [json.loads(line) for line in open(synth_manifest)]

    before = _gray_stats(rows[0]["image"])
    out_rows: list[dict[str, str]] = []
    for i, r in enumerate(rows):
        with Image.open(r["image"]) as im:
            g = np.asarray(im.convert("L"), dtype=np.float32)
        # per-image standardize to the real (mean, std): lightens ink + compresses contrast toward real scans
        std = max(g.std(), 1e-3)
        h = (g - g.mean()) / std * tgt_std + tgt_mean
        h = np.clip(h, 0.0, 255.0).astype(np.uint8)
        dst = os.path.join(img_dir, f"{i:05d}.png")
        Image.fromarray(h, mode="L").convert("RGB").save(dst)  # RGB so TrOCRProcessor is happy
        out_rows.append({"image": os.path.abspath(dst), "text": r["text"]})

    with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    after = _gray_stats(out_rows[0]["image"])
    print(f"target real (mean,std) = ({tgt_mean:.1f}, {tgt_std:.1f})")
    print(f"synth[0] before -> after:  mean {before[0]:.1f}->{after[0]:.1f}  "
          f"std {before[1]:.1f}->{after[1]:.1f}  ink {before[2]:.3f}->{after[2]:.3f}")
    print(f"harmonized {len(out_rows)} lines -> {out_dir}/train.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description="Harmonize synth line images to real-IAM photometrics.")
    ap.add_argument("--synth", default="runs/htr_ab/synth_fixed/train.jsonl")
    ap.add_argument("--real", default="runs/htr_ab/real/train.jsonl")
    ap.add_argument("--out", default="runs/htr_ab/synth_harmonized")
    args = ap.parse_args()
    harmonize(args.synth, args.real, args.out)


if __name__ == "__main__":
    main()
