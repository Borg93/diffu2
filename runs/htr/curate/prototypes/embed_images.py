"""Qwen3-VL-Embedding 8B vectors for the scans listed in a file, written as Parquet part files in a folder (image, qwen8b), resumable; the image alone,
fitted to 896 px, as embed_scans_semantic.py does. Runs in the scratch embedding environment, which has no Lance:
embeddings_to_lance.py moves the rows into scan_embeddings.lance.

    CUDA_VISIBLE_DEVICES=2 venv_egemma/bin/python embed_images.py IMAGES.txt OUT_FOLDER
"""
import sys, time
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None
images = [line.rstrip("\n") for line in open(sys.argv[1], encoding="utf-8") if line.strip()]; BATCH = 16


class Scans(Dataset):
    def __len__(self): return len(images)
    def __getitem__(self, i):
        try:
            with Image.open(images[i]) as scan:
                scan.draft("RGB", (1792, 1792)); scan = scan.convert("RGB"); scan.thumbnail((896, 896)); return i, scan
        except (OSError, ValueError):  # an unreadable or truncated scan
            return i, None


out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True); PART = 2000  # scans a part file; a stopped run loses less than one
done = {image for part in out_dir.glob("part-*.parquet") for image in pq.read_table(part, columns=["image"]).column("image").to_pylist()}
images = [image for image in images if image not in done]; print(len(done), "done before,", len(images), "to embed", flush=True)


def write(kept: list[str], vectors: list[np.ndarray]) -> None:
    stacked = np.concatenate(vectors)
    table = pa.table({"image": kept, "qwen8b": pa.FixedSizeListArray.from_arrays(pa.array(stacked.ravel(), pa.float16()), stacked.shape[1])})
    pq.write_table(table, out_dir / f"part-{int(time.time() * 1000)}.parquet")


model = SentenceTransformer("/mnt/scratch/gabriel/models/qwen3-vl-embedding-8B", device="cuda", model_kwargs={"dtype": torch.bfloat16})
kept, vectors, total, started = [], [], 0, time.time()
for n, batch in enumerate(DataLoader(Scans(), batch_size=BATCH, num_workers=12, collate_fn=lambda b: b, prefetch_factor=4)):
    batch = [(i, scan) for i, scan in batch if scan is not None]
    if not batch: continue
    out = model.encode([{"image": scan} for _, scan in batch], batch_size=BATCH, normalize_embeddings=True)
    kept += [images[i] for i, _ in batch]; vectors.append(np.asarray(out, np.float16)); total += len(batch)
    if len(kept) >= PART: write(kept, vectors); kept, vectors = [], []
    if n % 50 == 49: print(total, "scans, %.1f a second" % (total / (time.time() - started)), flush=True)
if kept: write(kept, vectors)
print("done", total, "of", len(images), flush=True)
