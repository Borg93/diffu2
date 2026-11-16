"""Scan embeddings into scan_embeddings.lance (image, qwen8b: 4,096 float16), from the files the embedders wrote.

The 15,000-scan run's arrays (embed_scans/sem_qwen8b.npy with its row and ok files) and any Parquet file given;
an image already in the table is left as it is.

    uv run python runs/htr/curate/prototypes/embeddings_to_lance.py [PARQUET or FOLDER ...]
"""
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HOME = Path("runs/htr/curate"); OUT = HOME / "scan_embeddings.lance"; E = HOME / "embed_scans"


def held() -> set[str]:
    return set(lance.dataset(str(OUT)).to_table(columns=["image"]).column("image").to_pylist()) if OUT.exists() else set()


def add(images: list[str], vectors: np.ndarray) -> int:
    known = held(); keep = [i for i, image in enumerate(images) if image not in known]
    if keep:
        rows = np.ascontiguousarray(vectors[keep], np.float16)
        table = pa.table({"image": [images[i] for i in keep], "qwen8b": pa.FixedSizeListArray.from_arrays(pa.array(rows.ravel(), pa.float16()), rows.shape[1])})
        lance.write_dataset(table, str(OUT), mode="append" if OUT.exists() else "create")
    return len(keep)


if __name__ == "__main__":
    if not OUT.exists():
        rows = np.load(E / "sem_rows.npy"); ok = np.load(E / "sem_qwen8b_ok.npy"); images = pd.read_json(E / "sample.json").iloc[rows].image.to_numpy()
        print("the 15,000-scan run:", add(images[ok].tolist(), np.load(E / "sem_qwen8b.npy")[ok]), "rows")
    for path in (part for given in sys.argv[1:] for part in (sorted(Path(given).glob('part-*.parquet')) if Path(given).is_dir() else [Path(given)])):
        table = pq.read_table(path); vectors = np.asarray(table.column("qwen8b").combine_chunks().flatten()).reshape(table.num_rows, -1)
        print(path, add(table.column("image").to_pylist(), vectors), "rows")
    print("table:", lance.dataset(str(OUT)).count_rows(), "scans")
