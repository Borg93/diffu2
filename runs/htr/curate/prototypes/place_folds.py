"""The fold of every described opening, placed at full resolution around the middle of the scan (CURATION E31).

Against 200 folds marked by hand the placer of cut_place.py, searching 6 % of the width to either side of the middle
and not around the gutter network's answer, finds a fold line in 87 % of scans and is then off by more than 1 % of
the width in 2 %. Here it runs over every scan the vision-language model calls an opening and that has no row yet,
appending to folds.lance (image, fold_x, fold_stands, fold_found) in batches; resumable.

With ``pool``: over every pool scan with labelled lines that the gutter network calls an opening (2.9 million; of the
described scans the head would cut, 98.4 % are among such), in an order shuffled once, so that the scans placed at
any moment are a random sample of the pools.

    PYTHONPATH=runs/htr/curate/prototypes uv run python runs/htr/curate/prototypes/place_folds.py [WORKERS] [pool]
"""
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lance
import pyarrow as pa

import cut_place

HOME = Path("runs/htr/curate"); OUT = HOME / "folds.lance"; BAND, BATCH = 0.06, 2000
ROW = pa.schema([("image", pa.utf8()), ("fold_x", pa.float32()), ("fold_stands", pa.float32()), ("fold_found", pa.bool_())])


def widen() -> None: cut_place.BAND = BAND
def place(image: str) -> tuple[float, float] | None: return cut_place.place(image, 0.5)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    described = "vl_layout IN ('opening_both_pages_written', 'opening_one_page_empty')"
    pool = "pool != 'gt' AND lines > 0 AND gutter_score >= 0.5"
    openings = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image"], filter=pool if "pool" in sys.argv[2:] else described).column("image").to_pylist()
    if "pool" in sys.argv[2:]:
        random.Random(0).shuffle(openings)
    done = set(lance.dataset(str(OUT)).to_table(columns=["image"]).column("image").to_pylist()) if OUT.exists() else set()
    todo = [image for image in openings if image not in done]; print(len(openings), "openings,", len(todo), "to place", flush=True)
    with ProcessPoolExecutor(workers, initializer=widen) as pool:
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]; found = list(pool.map(place, batch, chunksize=8))
            rows = [(image, *got) for image, got in zip(batch, found) if got is not None]
            table = pa.table({"image": [r[0] for r in rows], "fold_x": [r[1] for r in rows], "fold_stands": [r[2] for r in rows], "fold_found": [r[2] >= cut_place.CLEAR for r in rows]}, schema=ROW)
            lance.write_dataset(table, str(OUT), mode="append" if OUT.exists() else "create")
            print(start + len(batch), "of", len(todo), flush=True)
    lance.dataset(str(OUT)).optimize.compact_files()
    print("done", flush=True)
