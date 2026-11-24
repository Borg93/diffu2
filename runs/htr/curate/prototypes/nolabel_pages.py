"""The page sides the sets never built because the old pipeline put no line on them, where the writing layer sees writing.

A scan with a fold gives two pages; ``diffu-reader-set`` writes a page only when it holds a labelled line, so 345,595
sides were never cut. The per-side writing layer (E61) says which of them hold writing: this cuts those (``written`` of
the side above MIN) at the fold, fits them as the reader's processor does and writes a reader set without targets, so
the reader can read them (CURATION E70).

    uv run python runs/htr/curate/prototypes/nolabel_pages.py /mnt/scratch/gabriel/htr/nolabel_v1 [MIN [WORKERS]]
"""
import glob, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lance, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image

from diffu_page.page.nemotron_parse import fit_page

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def cut(row: dict) -> str | None:
    out = Path(row["page_image"])
    if out.exists():
        return row["id"]
    try:
        with Image.open(row["image"]) as scan:
            x = round(row["fold"] * scan.width)
            box = (0, 0, x, scan.height) if row["side"] == "left" else (x, 0, scan.width, scan.height)
            fit_page(scan.crop(box)).save(out, quality=92)
    except OSError:
        return None
    return row["id"]


if __name__ == "__main__":
    root = Path(sys.argv[1]); least = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    parts = glob.glob("/mnt/scratch/gabriel/htr/writing_pages/*.parquet")
    written = pa.concat_tables([pq.read_table(p, columns=["image", "written_left", "written_right"]) for p in parts]).to_pandas().drop_duplicates("image").set_index("image")
    pages = lance.dataset("runs/htr/curate/train_pages.lance").to_table(columns=["page", "set", "image", "pool", "volume", "scan", "side", "fold"]).to_pandas()
    scans = pages.groupby("image").agg(n=("side", "size"), page=("page", "first"), side=("side", "first"), fold=("fold", "first"), set=("set", "first"), pool=("pool", "first"), volume=("volume", "first"), scan=("scan", "first"))
    one = scans[(scans.n == 1) & scans.fold.notna()].join(written)
    one["missing"] = np.where(one.side == "left", "right", "left")
    one["written"] = np.where(one.side == "left", one.written_right, one.written_left)
    one = one[one.written > least].sort_index()
    (root / "pages_pages").mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"{r.page.rsplit('_', 1)[0]}_{r.missing}", "image": image, "fold": float(r.fold), "side": r.missing, "pool": r.pool, "volume": r.volume, "scan": r.scan, "sibling": r.page, "sibling_set": r.set, "written": float(r.written)} for image, r in one.iterrows()]
    for row in rows:
        row["page_image"] = str(root / "pages_pages" / f"{row['id']}.jpg")
    print(len(rows), "sides to cut", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        done = set(filter(None, pool.map(cut, rows, chunksize=64)))
    rows = [row for row in rows if row["id"] in done]
    with open(root / "pages.jsonl", "w", encoding="utf-8") as samples, open(root / "pages.meta.jsonl", "w", encoding="utf-8") as metas:
        for row in rows:
            samples.write(json.dumps({"messages": [{"role": "user", "content": "<image>" + PROMPT}, {"role": "assistant", "content": ""}], "images": [row["page_image"]]}) + "\n")
            metas.write(json.dumps({"id": row["id"], "source": "real", "collection": "", "volume": row["volume"], "scan": row["scan"], "side": row["side"], "split": "train"}) + "\n")
    pq.write_table(pa.Table.from_pylist(rows), root / "sides.parquet")
    print("built", len(rows), flush=True)
