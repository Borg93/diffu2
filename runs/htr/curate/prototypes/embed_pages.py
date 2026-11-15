"""Page embeddings on a GPU for the embedding comparison: a stratified sample of single pages (pool pages by pool,
ground-truth pages with their collection and volume as labels to score clusters against), embedded by DINOv3 ViT-L
(class token + mean patch token) and TIPSv2-L (class token + mean patch token), each page fitted to 448 px.

    CUDA_VISIBLE_DEVICES=1 python embed_pages.py OUT [POOL_PAGES GT_PAGES]
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
Image.MAX_IMAGE_PIXELS = None
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
import pandas as pd
LONG = 448  # a page is fitted to this many pixels on its longer side
sample = pd.read_json(OUT / "sample.json")  # written by the main environment from pages.lance
sample.drop(columns=["image"]).to_json(OUT / "pages.json", orient="records")
print("pages", len(sample), sample.pool.value_counts().to_dict(), flush=True)

class Pages(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        try:
            with Image.open(r["image"]) as im:
                im.draft("RGB", (2400, 2400)); k = im.width / r["scan_width"]
                page = im.convert("RGB").crop((round(r["x0"] * k), round(r["y0"] * k), round(r["x1"] * k), round(r["y1"] * k)))
            s = LONG / max(page.size)
            page = page.resize((max(28, round(page.width * s / 112) * 112), max(28, round(page.height * s / 112) * 112)), Image.BICUBIC)
            return i, torch.from_numpy(np.asarray(page)).permute(2, 0, 1).float() / 255
        except (OSError, ValueError):  # an unreadable or truncated scan; anything else is a bug and stops the run
            return i, None
def collate(b): return b
loader = DataLoader(Pages(sample.to_dict("records")), batch_size=1, num_workers=12, collate_fn=collate, prefetch_factor=8)
dino = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=torch.bfloat16).cuda().eval()
tips = AutoModel.from_pretrained("google/tipsv2-l14", trust_remote_code=True).vision_encoder.cuda().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(3, 1, 1)
D = np.zeros((len(sample), 2048), np.float16); T = np.zeros((len(sample), 2048), np.float16); ok = np.zeros(len(sample), bool)
t0 = time.time()
with torch.inference_mode():
    for n, batch in enumerate(loader):
        for i, x in batch:
            if x is None: continue
            x = x.cuda(); h, w = x.shape[1], x.shape[2]
            o = dino(pixel_values=((x - MEAN) / STD)[None].to(torch.bfloat16)).last_hidden_state[0].float(); p = (h // 16) * (w // 16)
            D[i] = torch.cat([F.normalize(o[0], dim=0), F.normalize(o[-p:].mean(0), dim=0)]).cpu().numpy()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, _, pt = tips(x[None])
            T[i] = torch.cat([F.normalize(c[0, 0].float(), dim=0), F.normalize(pt[0].float().mean(0), dim=0)]).cpu().numpy()
            ok[i] = True
        if n % 2000 == 1999: print(n + 1, "pages, %.1f a second" % ((n + 1) / (time.time() - t0)), flush=True)
np.save(OUT / "dinov3.npy", D); np.save(OUT / "tipsv2.npy", T); np.save(OUT / "ok.npy", ok)
print("done", int(ok.sum()), "embedded in %.0f s" % (time.time() - t0), flush=True)
