"""Where the pool pretraining helps on the 200 test pages (CURATION E44): by how like the pool's scans a test page is.

For every test page, its scan's Qwen 8B vector against those of the 75,776 pool scans the pretraining used: the mean
cosine similarity of its ten nearest. Then the reader's error without pool pages and after the trusted-cut
pretraining, by third of that likeness, and by whether the page is greyscale (microfilm, as most of the pools are) or
colour.

    uv run --with rapidfuzz python runs/htr/curate/prototypes/gain_by_likeness.py
"""
import json
from pathlib import Path

import lance
import numpy as np
import pandas as pd
from PIL import Image

from compare_reads import rate, rows

HOME = Path("runs/htr/curate")

if __name__ == "__main__":
    none, trusted = rows("test_read_norepeat10"), rows("test_read_trusted_norepeat10")
    meta = pd.DataFrame([json.loads(line) for line in open("data/htr/nemotron_parse_pages.test.meta.jsonl", encoding="utf-8")]).set_index("id").loc[none.index]
    scans = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", "volume", "scan"], filter="pool = 'gt'").to_pandas()
    scans["key"] = scans.volume + "/" + scans.scan
    image = meta.apply(lambda page: page["volume"] + "/" + Path(page["scan"]).stem, axis=1).map(scans.drop_duplicates("key").set_index("key").image)
    print(int(image.notna().sum()), "of", len(image), "test pages matched to their scan")
    used = set(pd.read_parquet("/mnt/scratch/gabriel/htr/trusted_v1/trusted.parquet").image)
    held = lance.dataset(str(HOME / "scan_embeddings.lance")).to_table().to_pandas()
    vectors = np.stack(held.qwen8b.to_numpy()).astype(np.float32); vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    place = {name: i for i, name in enumerate(held.image)}
    pool = vectors[[i for name, i in place.items() if name in used]]
    print(len(pool), "pool scans of the pretraining with a vector")
    like = pd.Series(np.nan, index=none.index)
    for page, name in image.dropna().items():
        if name in place:
            like[page] = np.sort(pool @ vectors[place[name]])[-10:].mean()
    def grey(page: str) -> bool:
        with Image.open(f"data/htr/nemotron_parse_pages_pages/{page}.jpg") as shown:
            shown = shown.convert("HSV"); shown.thumbnail((200, 200))
        return float(np.asarray(shown)[..., 1].mean()) < 12  # mean saturation of 255: a greyscale scan is near 0
    is_grey = pd.Series({page: grey(page) for page in none.index})
    known = like.dropna().index; third = pd.qcut(like[known], 3, labels=["least like the pools", "middle", "most like the pools"])
    for title, by in (("by likeness to the pool scans (mean cosine of the ten nearest)", third), ("by scan colour", is_grey.map({True: "greyscale", False: "colour"}))):
        table = pd.DataFrame({"pages": by.value_counts(), "chars": none.loc[by.index].groupby(by.to_numpy()).chars.sum(),
                              "none": none.loc[by.index].groupby(by.to_numpy()).apply(rate), "trusted": trusted.loc[by.index].groupby(by.to_numpy()).apply(rate)})
        table["gain, points"] = table.none - table.trusted; table["gain, relative"] = 100 * table["gain, points"] / table.none
        print(f"\n== {title}"); print(table.round(2).to_string())
    print("\nlikeness by third:", like[known].groupby(third.to_numpy()).agg(["min", "max"]).round(3).to_dict("index"))
    gain = ((none.edits - trusted.edits) / none.chars.clip(lower=1)).loc[known]
    print("rank correlation of a page's gain with its likeness:", round(float(gain.corr(like[known], method="spearman")), 3))
