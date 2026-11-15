"""EmbeddingGemma 2 page embeddings of the same sample: the page image alone, and the page image with its label's text
interleaved (clustering prompt), each 768-d, normalised."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
Image.MAX_IMAGE_PIXELS = None
OUT = Path(sys.argv[1]); s = pd.read_json(OUT / "sample.json"); texts = json.load(open(OUT / "texts.json"))
class Pages(Dataset):
    def __len__(self): return len(s)
    def __getitem__(self, i):
        r = s.iloc[i]
        try:
            with Image.open(r["image"]) as im:
                im.draft("RGB", (1600, 1600)); k = im.width / r["scan_width"]
                page = im.convert("RGB").crop((round(r["x0"] * k), round(r["y0"] * k), round(r["x1"] * k), round(r["y1"] * k)))
            page.thumbnail((896, 896)); return i, page
        except Exception:
            return i, None
loader = DataLoader(Pages(), batch_size=32, num_workers=12, collate_fn=lambda b: b, prefetch_factor=4)
m = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"dtype": torch.bfloat16}, config_kwargs={"audio_config": None})
A = np.zeros((len(s), 768), np.float16); B = np.zeros((len(s), 768), np.float16); ok = np.zeros(len(s), bool); t0 = time.time()
for n, batch in enumerate(loader):
    batch = [(i, p) for i, p in batch if p is not None]
    if not batch: continue
    idx = [i for i, _ in batch]
    a = m.encode([{"text": "task: clustering | query: <|image|>", "image": [p]} for _, p in batch], batch_size=32, normalize_embeddings=True)
    b = m.encode([{"text": "task: clustering | query: <|image|> " + texts[i][:1500].replace("\n", " "), "image": [p]} for i, p in batch], batch_size=32, normalize_embeddings=True)
    A[idx] = a; B[idx] = b; ok[idx] = True
    if n % 100 == 99: print((n + 1) * 32, "pages, %.1f a second" % ((n + 1) * 32 / (time.time() - t0)), flush=True)
np.save(OUT / "egemma_image.npy", A); np.save(OUT / "egemma_image_text.npy", B); np.save(OUT / "egemma_ok.npy", ok)
print("done", int(ok.sum()), "in %.0f s" % (time.time() - t0), flush=True)
