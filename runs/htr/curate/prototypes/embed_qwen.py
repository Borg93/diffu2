"""Qwen3-VL-Embedding page embeddings of the same sample as embed_gemma.py, the same way: the page image alone, and
the page image with its label's text, each page fitted to 896 px, through Sentence Transformers with the model's own
default instruction ("Represent the user's input."), normalised. The model is read from a folder on scratch that
links to the Hub cache's weights and adds the pooling config the cache lacks (the cache folder is root's).

    CUDA_VISIBLE_DEVICES=2 python embed_qwen.py OUT 2B|8B [LIMIT]
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
Image.MAX_IMAGE_PIXELS = None
OUT, SIZE = Path(sys.argv[1]), sys.argv[2]; LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
s = pd.read_json(OUT / "sample.json"); texts = json.load(open(OUT / "texts.json")); BATCH = 16
class Pages(Dataset):
    def __len__(self): return LIMIT or len(s)
    def __getitem__(self, i):
        r = s.iloc[i]
        try:
            with Image.open(r["image"]) as im:
                im.draft("RGB", (1600, 1600)); k = im.width / r["scan_width"]
                page = im.convert("RGB").crop((round(r["x0"] * k), round(r["y0"] * k), round(r["x1"] * k), round(r["y1"] * k)))
            page.thumbnail((896, 896)); return i, page
        except (OSError, ValueError):  # an unreadable or truncated scan; anything else is a bug and stops the run
            return i, None
loader = DataLoader(Pages(), batch_size=BATCH, num_workers=12, collate_fn=lambda b: b, prefetch_factor=4)
m = SentenceTransformer(f"/mnt/scratch/gabriel/models/qwen3-vl-embedding-{SIZE.upper()}", device="cuda", model_kwargs={"dtype": torch.bfloat16})
A = B = None; ok = np.zeros(len(s), bool); t0 = time.time()
for n, batch in enumerate(loader):
    batch = [(i, p) for i, p in batch if p is not None]
    if not batch: continue
    idx = [i for i, _ in batch]
    a = m.encode([{"image": p} for _, p in batch], batch_size=BATCH, normalize_embeddings=True)
    b = m.encode([{"text": texts[i][:1500].replace("\n", " "), "image": p} for i, p in batch], batch_size=BATCH, normalize_embeddings=True)
    if A is None: A = np.zeros((len(s), a.shape[1]), np.float16); B = np.zeros((len(s), a.shape[1]), np.float16)
    A[idx] = a; B[idx] = b; ok[idx] = True
    if n % 100 == 99 or LIMIT: print((n + 1) * BATCH, "pages, %.1f a second" % ((n + 1) * BATCH / (time.time() - t0)), "dim", a.shape[1], flush=True)
tag = f"qwen{SIZE.lower()}" + ("_smoke" if LIMIT else "")
np.save(OUT / f"{tag}_image.npy", A); np.save(OUT / f"{tag}_image_text.npy", B); np.save(OUT / f"{tag}_ok.npy", ok)
print("done", int(ok.sum()), "in %.0f s" % (time.time() - t0), flush=True)
