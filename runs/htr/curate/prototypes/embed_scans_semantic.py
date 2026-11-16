"""One vector a scan from the image-and-text embedders, for clustering the described scans (CURATION E21).

The scans are a fixed random 15,000 of the 45,000 the vision-language model described (runs/htr/curate/embed_scans,
whose TIPSv2 and DINOv3 vectors exist already), each whole, fitted to 896 px, the image alone: no label text, so a
cluster says what a scan looks like and is, not what its old label reads.

    CUDA_VISIBLE_DEVICES=2 python embed_scans_semantic.py nvembed8b|qwen8b|egemma2 [LIMIT]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

Image.MAX_IMAGE_PIXELS = None
E = Path("runs/htr/curate/embed_scans"); NAME = sys.argv[1]; LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
sample = pd.read_json(E / "sample.json"); COUNT, BATCH = 15000, 16
which = np.random.default_rng(0).permutation(len(sample))[:COUNT]; np.save(E / "sem_rows.npy", which)


class Scans(Dataset):
    def __len__(self): return LIMIT or len(which)
    def __getitem__(self, i):
        try:
            with Image.open(sample.image.iloc[which[i]]) as scan:
                scan.draft("RGB", (1792, 1792)); scan = scan.convert("RGB"); scan.thumbnail((896, 896)); return i, scan
        except (OSError, ValueError):  # an unreadable or truncated scan; anything else is a bug and stops the run
            return i, None


if NAME == "nvembed8b":
    model = SentenceTransformer("/mnt/scratch/gabriel/models/nemotron-3.5-embed-8b", device="cuda", model_kwargs={"dtype": torch.bfloat16})
    def embed(images): return model.encode_document([{"image": image} for image in images], batch_size=BATCH, normalize_embeddings=True)
elif NAME == "qwen8b":
    model = SentenceTransformer("/mnt/scratch/gabriel/models/qwen3-vl-embedding-8B", device="cuda", model_kwargs={"dtype": torch.bfloat16})
    def embed(images): return model.encode([{"image": image} for image in images], batch_size=BATCH, normalize_embeddings=True)
elif NAME == "egemma2":
    model = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"dtype": torch.bfloat16}, config_kwargs={"audio_config": None})
    def embed(images): return model.encode([{"text": "task: clustering | query: <|image|>", "image": [image]} for image in images], batch_size=BATCH, normalize_embeddings=True)
else:
    raise SystemExit(f"no embedder named {NAME}")
vectors = None; ok = np.zeros(len(which), bool); started = time.time()
for n, batch in enumerate(DataLoader(Scans(), batch_size=BATCH, num_workers=12, collate_fn=lambda b: b, prefetch_factor=4)):
    batch = [(i, scan) for i, scan in batch if scan is not None]
    if not batch: continue
    out = embed([scan for _, scan in batch])
    if vectors is None: vectors = np.zeros((len(which), out.shape[1]), np.float16)
    vectors[[i for i, _ in batch]] = out; ok[[i for i, _ in batch]] = True
    if n % 50 == 49 or LIMIT: print((n + 1) * BATCH, "scans, %.1f a second" % ((n + 1) * BATCH / (time.time() - started)), "dim", out.shape[1], flush=True)
tag = NAME + ("_smoke" if LIMIT else "")
np.save(E / f"sem_{tag}.npy", vectors); np.save(E / f"sem_{tag}_ok.npy", ok); print("done", int(ok.sum()), "in %.0f s" % (time.time() - started), flush=True)
