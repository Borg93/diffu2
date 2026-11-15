"""Prototype: a small CNN on a greyscale thumbnail of a scan -> per-column "the gutter is here". Trained on openings
whose two pages both hold labelled lines (the cut between their text blocks, from the page table) and on single
sheets (no gutter). CPU only."""
import sys, json, random, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import lance
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
W, H, COLS = 384, 256, 96  # thumbnail size, and the columns the answer is given on
torch.set_num_threads(16); random.seed(0); torch.manual_seed(0)

def thumb(path):
    try:
        with Image.open(path) as im:
            im.draft("L", (W * 2, H * 2)); return np.asarray(im.convert("L").resize((W, H), Image.BILINEAR), dtype=np.uint8)
    except Exception:
        return None

d = lance.dataset("runs/htr/curate/pages.lance")
cols = ["pool", "scan", "side", "image", "scan_width", "scan_height", "x1", "lines", "alone", "spread", "split"]
t = d.to_table(columns=cols, filter="error IS NULL").to_pandas()
t = t[t.split != "test"]
both = t[(t.side == "left") & (t.alone == False)]  # the left page's right edge is the cut between two labelled pages
both = both.assign(y=both.x1 / both.scan_width)
single = t[(t.side == "whole") & (t.lines > 0) & (t.scan_width < 0.8 * t.scan_height)].assign(y=-1.0)
def take(df, pool, n): return df[df.pool == pool].sample(min(n, (df.pool == pool).sum()), random_state=0)
pos = [take(both, p, n) for p, n in (("gt", 5000), ("svea", 4000), ("andraarkiv", 4000))]
neg = [take(single, p, n) for p, n in (("gt", 2000), ("svea", 1500), ("andraarkiv", 2500))]
import pandas as pd
data = pd.concat(pos + neg).sample(frac=1, random_state=0).reset_index(drop=True)
print("scans:", data.groupby(["pool", data.y >= 0]).size().to_dict(), flush=True)
t0 = time.time()
with Pool(24) as pool: images = pool.map(thumb, data.image.tolist(), chunksize=32)
keep = [i for i, im in enumerate(images) if im is not None]
X = torch.from_numpy(np.stack([images[i] for i in keep])).float().div(255).unsqueeze(1); data = data.iloc[keep].reset_index(drop=True)
print("thumbnails", len(X), "in %.0f s" % (time.time() - t0), flush=True)
y = torch.tensor(data.y.values, dtype=torch.float32)
target = torch.zeros(len(y), COLS); centres = (torch.arange(COLS) + 0.5) / COLS
has = y >= 0; target[has] = torch.exp(-0.5 * ((centres[None] - y[has, None]) / 0.012) ** 2)
n_test = 1500; Xtr, Ttr, Xte, Tte, yte, meta = X[n_test:], target[n_test:], X[:n_test], target[:n_test], y[:n_test], data.iloc[:n_test]

def block(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.a, self.b, self.c, self.d = block(1, 16), block(16, 32), block(32, 64), block(64, 96)
        self.head = nn.Conv1d(96, 1, 5, padding=2)
    def forward(self, x):  # [B,1,H,W] -> [B,COLS]
        x = F.max_pool2d(self.a(x), 2); x = F.max_pool2d(self.b(x), 2); x = self.c(x); x = self.d(F.max_pool2d(x, (2, 1)))
        return self.head(x.mean(2)).squeeze(1)  # rows pooled away: [B,96,W/4] -> [B,COLS]
net = Net(); print("parameters %.2f M" % (sum(p.numel() for p in net.parameters()) / 1e6), flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); EPOCHS = 8
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * (len(Xtr) // 64 + 1))
def score(Xs):
    net.eval(); out = []
    with torch.inference_mode():
        for i in range(0, len(Xs), 256): out.append(torch.sigmoid(net(Xs[i:i + 256])))
    p = torch.cat(out); lo, hi = int(0.15 * COLS), int(0.85 * COLS); s, j = p[:, lo:hi].max(1); return s, (j + lo + 0.5) / COLS
for epoch in range(EPOCHS):
    net.train(); perm = torch.randperm(len(Xtr)); total = 0.0
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]; xb = Xtr[idx]
        xb = xb * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, 1)) + 0.1 * (torch.rand(len(idx), 1, 1, 1) - 0.5)  # exposure
        loss = F.binary_cross_entropy_with_logits(net(xb), Ttr[idx], pos_weight=torch.tensor(8.0)); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); total += loss.item()
    s, p = score(Xte); o = yte >= 0; err = (p[o] - yte[o]).abs()
    print("epoch %d loss %.4f | openings: within 1%% of width %.3f, within 2%% %.3f, score>0.5 %.3f | single sheets score>0.5 %.3f" % (epoch, total / (len(perm) / 64), (err < .01).float().mean(), (err < .02).float().mean(), (s[o] > .5).float().mean(), (s[~o] > .5).float().mean()), flush=True)
torch.save(net.state_dict(), OUT / "small_gutter.pt")
s, p = score(Xte); o = (yte >= 0).numpy(); err = (p - yte).abs().numpy(); s = s.numpy()
for pool in ("gt", "svea", "andraarkiv"):
    m = (meta.pool == pool).values
    print(f"{pool}: openings {int((m & o).sum())}: within 2% {np.mean(err[m & o] < .02):.3f}, median off {np.median(err[m & o]):.4f}, missed (score<0.5) {np.mean(s[m & o] < .5):.3f} | single sheets {int((m & ~o).sum())}: false alarm {np.mean(s[m & ~o] > .5):.3f}", flush=True)
t0 = time.time(); score(Xte[:512]); print("inference %.1f ms a scan on CPU (16 threads), thumbnail aside" % ((time.time() - t0) / 512 * 1000), flush=True)
json.dump({"test": [{"image": r.image, "pool": r.pool, "y": float(r.y), "score": float(a), "pos": float(b)} for r, a, b in zip(meta.itertuples(), s, p.numpy())]}, open(OUT / "test.json", "w"))
print("done", flush=True)
