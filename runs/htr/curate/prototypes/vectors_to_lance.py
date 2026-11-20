"""The DINOv3 and TIPSv2 vectors of every embedded scan as columns of scans.lance (CURATION E40; a column asked for).

    flock runs/htr/curate/scans.lock uv run python runs/htr/curate/prototypes/vectors_to_lance.py

From embed_pool.py's parts (the pools, the ground truth): `dinov3` and `tipsv2`, each 2,048 16-bit numbers (the class
token and the mean patch token, each of unit length), about 13 GB a column. Run alone on the table; it is tagged first.
"""
import os
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0"); os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import time
from pathlib import Path
import lance, pyarrow as pa, pyarrow.parquet as pq
from diffu_curate.table import add_features

HOME = Path("runs/htr/curate"); PARTS = [Path("/mnt/scratch/gabriel/htr/embed_pool"), Path("/mnt/scratch/gabriel/htr/embed_gt")]
if __name__ == "__main__":
    start = time.time(); held = lance.dataset(str(HOME / "scans.lance"))
    assert "dinov3" not in held.schema.names, "the table holds the vectors already"
    if "before-vectors" not in held.tags.list(): held.tags.create("before-vectors", held.version)
    parts = sorted(part for folder in PARTS for part in folder.glob("part-*.parquet"))
    features = pa.concat_tables([pq.read_table(part, columns=["image", "dinov3", "tipsv2"]) for part in parts])
    print(f"{len(features):,} scans from {len(parts)} parts read in {time.time() - start:.0f} s, {features.nbytes / 1e9:.1f} GB; joining", flush=True); joining = time.time()
    print(f"{add_features(HOME / 'scans.lance', features):,} matched; joined in {time.time() - joining:.0f} s", flush=True)
    held = lance.dataset(str(HOME / "scans.lance")); print("version", held.version, held.schema.field("dinov3").type, "| rows with dinov3:", held.count_rows("dinov3 IS NOT NULL"), flush=True)
