"""Our reader against the old label, a page (CURATION E59): the disagreement of every page a vLLM pool read holds.

    uv run --with rapidfuzz python runs/htr/curate/prototypes/read_disagreement.py READ.jsonl OUT.parquet

For each page: the edits between the old label's text and the reader's (NFC, ends trimmed) over the label's characters
(``disagreement``), the two lengths, how the reading ended (``finish``), and the pool. Printed: the distribution, by
pool, and how many pages each threshold keeps. Needs no network disk.
"""
import json, sys, unicodedata
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein

def one(line):
    r = json.loads(line); t, p = (unicodedata.normalize("NFC", r[k]).strip() for k in ("text", "prediction"))
    return r["id"], r["subset"], len(t), len(p), Levenshtein.distance(t, p), r.get("finish"), r["tokens"]

if __name__ == "__main__":
    lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
    with ProcessPoolExecutor(16) as pool: rows = list(pool.map(one, lines, chunksize=2000))
    d = pd.DataFrame(rows, columns=["id", "volume", "label_chars", "read_chars", "edits", "finish", "tokens"]).drop_duplicates("id")
    d["disagreement"] = d.edits / d.label_chars.clip(lower=1); d["pool"] = np.where(d.volume.str.match(r"^[A-Z]\d{7}$"), "svea", "andraarkiv")
    d.to_parquet(sys.argv[2]); ok = d[d.finish == "stop"]
    print(len(d), "pages;", d.finish.value_counts().to_dict(), "| corpus disagreement of the pages that ended by themselves: %.2f %%" % (100 * ok.edits.sum() / ok.label_chars.sum()))
    print("quantiles of a page's disagreement:", {q: round(float(ok.disagreement.quantile(q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)})
    for pool, part in ok.groupby("pool"): print(f"{pool:11} {len(part):7} pages, median {part.disagreement.median():.3f}, corpus {100 * part.edits.sum() / part.label_chars.sum():.2f} %")
    for limit in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5): print(f"disagreement up to {limit:4}: keeps {int((ok.disagreement <= limit).sum()):7} pages ({(ok.disagreement <= limit).mean():.1%} of those that ended, {(d.finish.eq('stop') & (d.disagreement <= limit)).mean():.1%} of all)")
    print("label much longer than the reading (reading under 60 % of it):", int((ok.read_chars < 0.6 * ok.label_chars).sum()), "| reading much longer (label under 60 %):", int((ok.label_chars < 0.6 * ok.read_chars).sum()), "| label empty or under 20 characters:", int((d.label_chars < 20).sum()))
