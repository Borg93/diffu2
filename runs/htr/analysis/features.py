"""One row a line of the MiniCPM-OCR line set (train, dev, test): its meta, its text and what its crop shows.

Image measures are taken inside the line's polygon (the crop is white outside it): the paper's level, ink (pixels
clearly darker than the paper), where the ink starts and ends across the width, how many separate bands of ink it
makes down the height (2 or more: a second line in the crop), and a 128-bit average hash to find duplicate crops.

    python features.py OUT.parquet WORKERS
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SETS = Path("data/htr")
SPLITS = {"train": "minicpm_lines_scaled.train", "dev": "minicpm_lines_scaled.dev", "test": "minicpm_lines_scaled.test"}


def rows(split: str, stem: str) -> list[dict]:
    """The split's meta rows with each sample's text and image path (the two files are written row beside row)."""
    out = []
    with (SETS / f"{stem}.meta.jsonl").open(encoding="utf-8") as meta, (SETS / f"{stem}.jsonl").open(
        encoding="utf-8"
    ) as samples:
        for m, s in zip(meta, samples, strict=True):
            row, sample = json.loads(m), json.loads(s)
            row["split"] = split
            row["text"] = sample["messages"][-1]["content"]
            row["image"] = sample["images"][0]
            out.append(row)
    return out


def measure(path: str) -> dict:
    try:
        image = Image.open(path)
        w, h = image.size
        gray = np.asarray(image.convert("L"), dtype=np.int16)
    except Exception as error:  # a broken crop is itself a finding
        return {"w": -1, "h": -1, "error": str(error)[:200]}
    inside = gray < 250
    n_inside = int(inside.sum())
    result = {"w": w, "h": h, "inside_frac": n_inside / gray.size}
    if n_inside < 20:
        return result | {"ink_frac": 0.0, "bands": 0}
    # the paper's level: the median of what is under 250 (the mask is white), except on white paper, where what is
    # under 250 is few pixels and dark (the ink itself) and the paper is white too
    paper = float(np.median(gray[inside]))
    if n_inside < 0.3 * gray.size and paper < 160:
        paper = 255.0
    ink = gray < paper - 45
    n_ink = int(ink.sum())
    cols = ink.sum(axis=0) >= 2
    rows_ = ink.sum(axis=1) >= max(2, w // 200)
    xs = np.flatnonzero(cols)
    ys = np.flatnonzero(rows_)
    bands, gap, run = 0, 0, False
    min_gap = max(3, h // 8)
    for on in rows_:
        if on:
            if not run and (bands == 0 or gap >= min_gap):
                bands += 1
            run, gap = True, 0
        else:
            run = False
            gap += 1
    small = np.asarray(image.convert("L").resize((16, 8), Image.Resampling.BILINEAR), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    return result | {
        "paper": paper,
        "ink_frac": n_ink / gray.size,
        "contrast": paper - float(gray[ink].mean()) if n_ink else 0.0,
        "ink_x0": xs[0] / w if xs.size else -1.0,
        "ink_x1": (xs[-1] + 1) / w if xs.size else -1.0,
        "ink_cols_frac": xs.size / w,
        "ink_y0": ys[0] / h if ys.size else -1.0,
        "ink_y1": (ys[-1] + 1) / h if ys.size else -1.0,
        "bands": bands,
        "ahash": int("".join("1" if b else "0" for b in bits), 2).to_bytes(16, "big").hex(),
    }


def main() -> None:
    out, workers = Path(sys.argv[1]), int(sys.argv[2])
    table = [row for split, stem in SPLITS.items() for row in rows(split, stem)]
    print(f"{len(table)} lines", flush=True)
    with Pool(workers) as pool:
        measures = []
        for i, m in enumerate(pool.imap(measure, (r["image"] for r in table), chunksize=256)):
            measures.append(m)
            if i % 100_000 == 0:
                print(f"{i} measured", flush=True)
    frame = pd.DataFrame(table).join(pd.DataFrame(measures))
    frame.to_parquet(out, index=False)
    print(f"{len(frame)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
