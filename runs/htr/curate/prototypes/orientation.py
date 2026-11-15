"""Prototype: which way up is the writing on a single page? A small CNN on a square greyscale thumbnail, trained on
pages from the page table (both sides labelled, so the cut is sound) turned by 0/90/180/270 degrees on purpose. Square
input, so the page's shape cannot give the answer away. CPU only."""
import sys, json, random, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import lance, pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SIDE = 256
torch.set_num_threads(10); random.seed(0); torch.manual_seed(0); np.random.seed(0)

def page_thumb(job):
    path, fx0, fx1 = job
    try:
        with Image.open(path) as im:
            im.draft("L", (SIDE * 4, SIDE * 4)); im = im.convert("L")
            page = im.crop((round(fx0 * im.width), 0, round(fx1 * im.width), im.height))
            return np.asarray(page.resize((SIDE, SIDE), Image.BILINEAR), dtype=np.uint8)
    except Exception:
        return None

d = lance.dataset("runs/htr/curate/pages.lance")
t = d.to_table(columns=["page", "pool", "side", "image", "scan_width", "x0", "x1", "lines", "alone", "split"], filter="error IS NULL AND lines >= 8").to_pandas()
t = t[(t.side != "whole") & (t.alone == False) & (t.split != "test")]
data = pd.concat([t[t.pool == p].sample(n, random_state=0) for p, n in (("gt", 5000), ("svea", 4500), ("andraarkiv", 4500))]).sample(frac=1, random_state=0).reset_index(drop=True)
jobs = [(r.image, r.x0 / r.scan_width, r.x1 / r.scan_width) for r in data.itertuples()]
t0 = time.time()
with Pool(16) as pool: images = pool.map(page_thumb, jobs, chunksize=32)
keep = [i for i, im in enumerate(images) if im is not None]; data = data.iloc[keep].reset_index(drop=True)
X = torch.from_numpy(np.stack([images[i] for i in keep])).float().div(255).unsqueeze(1)
print("pages", len(X), data.pool.value_counts().to_dict(), "thumbnails in %.0f s" % (time.time() - t0), flush=True)
n_test = 1600; Xtr, Xte, meta = X[n_test:], X[:n_test], data.iloc[:n_test]

def block(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
net = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 96), block(96, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 4))
print("parameters %.2f M" % (sum(p.numel() for p in net.parameters()) / 1e6), flush=True)
def turned(x, k): return torch.stack([torch.rot90(img, int(q), (1, 2)) for img, q in zip(x, k)])
EPOCHS = 6; opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * (len(Xtr) // 64 + 1))
kte = torch.arange(len(Xte)) % 4; Xte_t = turned(Xte, kte)
def predict(x):
    net.eval(); out = []
    with torch.inference_mode():
        for i in range(0, len(x), 256): out.append(F.softmax(net(x[i:i + 256]), 1))
    return torch.cat(out)
for epoch in range(EPOCHS):
    net.train(); perm = torch.randperm(len(Xtr)); total = 0.0
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]; k = torch.randint(0, 4, (len(idx),)); xb = turned(Xtr[idx], k)
        xb = xb * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, 1)) + 0.1 * (torch.rand(len(idx), 1, 1, 1) - 0.5)
        loss = F.cross_entropy(net(xb), k); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); total += loss.item()
    p = predict(Xte_t); acc = (p.argmax(1) == kte).float().mean().item()
    print("epoch %d loss %.3f | held-out accuracy %.4f" % (epoch, total / (len(perm) / 64), acc), flush=True)
torch.save(net.state_dict(), OUT / "orientation.pt")
p = predict(Xte_t); pred = p.argmax(1)
conf = torch.zeros(4, 4, dtype=torch.int64)
for a, b in zip(kte, pred): conf[a, b] += 1
print("confusion (rows: true turn 0/90/180/270, columns: predicted):", conf.tolist(), flush=True)
for pool in ("gt", "svea", "andraarkiv"):
    m = torch.tensor((meta.pool == pool).values); print(pool, "accuracy %.4f" % (pred[m] == kte[m]).float().mean().item(), "n", int(m.sum()), flush=True)
# untouched pages: what does it say of pages as they are? (those not called upright are either turned in truth or mistakes)
p0 = predict(Xte); up = (p0.argmax(1) == 0)
print("pages as they are: called upright %.4f; called turned: %d of %d" % (up.float().mean().item(), int((~up).sum()), len(Xte)), flush=True)
json.dump([{"page": meta.page.iloc[i], "image": meta.image.iloc[i], "pred": int(p0[i].argmax()), "conf": float(p0[i].max())} for i in torch.nonzero(~up).flatten().tolist()], open(OUT / "called_turned.json", "w"))
t0 = time.time(); predict(Xte[:512]); print("inference %.2f ms a page on CPU" % ((time.time() - t0) / 512 * 1000), flush=True)
print("done", flush=True)
