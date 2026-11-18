"""Test reads compared page by page (CURATION E38).

Each run's readings of the 200 test pages scored as htr-eval's OpenBMB figures are (NFC, the ends trimmed; edits
summed over the truth's characters, or words), then run against run: pages better and worse, the difference with a
bootstrap interval over pages, by the pool a page's volume is in, by kind of page (e22_rows.json), without the two
pages whose truth is partial, and the pages that moved most.

    uv run --with rapidfuzz python runs/htr/curate/prototypes/compare_reads.py NAME=FOLDER ... [BASE=NAME ...]

FOLDER is below runs/htr/nemotron_parse; the first run named is compared with every other.
"""
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from diffu_curate.settings import CurateSettings, volume_of

READS = Path("runs/htr/nemotron_parse"); PARTIAL = ["e6e45ddcae8c_right", "4c7c6e97ab16_left"]


def rows(folder: str) -> pd.DataFrame:
    out = []
    for line in open(READS / folder / "pages.jsonl", encoding="utf-8"):
        page = json.loads(line)
        truth, read = (unicodedata.normalize("NFC", page[key]).strip() for key in ("text", "prediction"))
        out.append({"id": page["id"], "subset": page["subset"], "edits": Levenshtein.distance(truth, read), "chars": len(truth),
                    "word_edits": Levenshtein.distance(truth.split(), read.split()), "words": len(truth.split())})
    return pd.DataFrame(out).set_index("id")


def rate(pages: pd.DataFrame, what: str = "edits", of: str = "chars") -> float:
    return 100 * pages[what].sum() / max(pages[of].sum(), 1)


if __name__ == "__main__":
    runs = {name: rows(folder) for name, folder in (arg.split("=") for arg in sys.argv[1:])}
    first = next(iter(runs)); ids = runs[first].index
    assert all(set(run.index) == set(ids) for run in runs.values())
    meta = {json.loads(line)["id"]: json.loads(line) for line in open("data/htr/nemotron_parse_pages.test.meta.jsonl", encoding="utf-8")}
    settings = CurateSettings()
    def pool_of(page: str) -> str:
        volume = volume_of(Path(meta[page]["scan"]).stem)
        return next((name for name, pool in settings.pools.items() if (pool.labels / volume).is_dir()), "in no pool")
    group = pd.Series({page: pool_of(page) for page in ids})
    kinds = pd.read_json("runs/htr/curate/e22_rows.json").drop_duplicates("id").set_index("id").writing.reindex(ids)

    print("== all 200 pages"); print(pd.DataFrame({name: {"CER": rate(run), "WER": rate(run, "word_edits", "words")} for name, run in runs.items()}).round(2).to_string())
    print("\n== without the two pages whose truth is partial"); print(pd.Series({name: rate(run.drop(PARTIAL)) for name, run in runs.items()}).round(2).to_string())
    for title, by in (("by the pool the page's volume is in", group), ("by kind of page (VL model)", kinds)):
        table = pd.DataFrame({name: run.loc[ids].groupby(by.to_numpy()).apply(rate) for name, run in runs.items()}).round(2)
        table.insert(0, "pages", by.value_counts()); table.insert(1, "chars", runs[first].loc[ids].groupby(by.to_numpy()).chars.sum())
        print(f"\n== {title}"); print(table.to_string())
    rng = np.random.default_rng(0); new = runs[first].loc[ids]
    for name, run in runs.items():
        if name == first:
            continue
        old = run.loc[ids]; better, worse = int((new.edits < old.edits).sum()), int((new.edits > old.edits).sum())
        draws = rng.integers(0, len(ids), (10000, len(ids)))
        gain = 100 * (old.edits.to_numpy()[draws].sum(1) - new.edits.to_numpy()[draws].sum(1)) / new.chars.to_numpy()[draws].sum(1)
        print(f"\n== {first} against {name}: {rate(old):.2f} -> {rate(new):.2f} %, gain {rate(old) - rate(new):+.2f} points "
              f"(95 % {np.percentile(gain, 2.5):+.2f} to {np.percentile(gain, 97.5):+.2f}; share of draws with a gain {np.mean(gain > 0):.3f}); "
              f"better on {better} pages, worse on {worse}, the same on {len(ids) - better - worse}")
        moved = (old.edits - new.edits).sort_values()
        for label, part in (("most worse", moved.head(5)), ("most better", moved.tail(5)[::-1])):
            print(f"   {label}: " + "; ".join(f"{page} {int(-change):+d} edits ({kinds[page]}, {group[page]})" for page, change in part.items()))
