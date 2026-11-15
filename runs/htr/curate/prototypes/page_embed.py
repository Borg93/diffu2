"""DINOv3 page embeddings of a random sample of single pages: do typed cards, placeholder sheets, blanks, ledgers and
prose fall into clusters one could select or drop, and how many near-duplicates are there? CPU."""
import sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import lance, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from transformers import AutoModel
Image.MAX_IMAGE_PIXELS = None; torch.set_num_threads(20)
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LONG, P, K = 448, 16, 30
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
d = lance.dataset("runs/htr/curate/pages.lance")
t = d.to_table(columns=["page", "pool", "scan", "side", "image", "scan_width", "scan_height", "x0", "y0", "x1", "y1", "lines", "alone", "spread"], filter="error IS NULL AND pool != 'gt'").to_pandas()
parts = [t[(t.lines >= 5) & (t.spread == False)].sample(2200, random_state=0), t[(t.lines > 0) & (t.lines < 5)].sample(400, random_state=0), t[t.lines == 0].sample(400, random_state=0)]
known = t[t.scan.isin(["C0025324_00001", "C0075198_00001", "C0074851_00001", "40008086_00079", "40008217_00266", "A0038868_00207", "C0074413_00002"])]
data = pd.concat(parts + [known]).drop_duplicates("page").reset_index(drop=True); print("pages", len(data), flush=True)
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").eval()
def page_image(r):
    with Image.open(r.image) as im:
        im.draft("RGB", (2000, 2000)); k = im.width / r.scan_width
        return im.convert("RGB").crop((round(r.x0 * k), round(r.y0 * k), round(r.x1 * k), round(r.y1 * k)))
@torch.inference_mode()
def embed(img):
    k = LONG / max(img.size); a = img.resize((max(P, round(img.width * k / P) * P), max(P, round(img.height * k / P) * P)), Image.BICUBIC)
    x = (torch.from_numpy(np.asarray(a)).permute(2, 0, 1).float() / 255 - MEAN) / STD
    t_ = model(pixel_values=x[None]).last_hidden_state[0]; n = (a.height // P) * (a.width // P)
    return torch.cat([F.normalize(t_[0], dim=0), F.normalize(t_[-n:].mean(0), dim=0)]).numpy()
E_, thumbs, t0 = [], [], time.time()
for i, r in enumerate(data.itertuples()):
    try:
        img = page_image(r); E_.append(embed(img)); th = img.copy(); th.thumbnail((150, 150)); thumbs.append(th)
    except Exception as e:
        E_.append(None); thumbs.append(None)
    if i % 500 == 499: print(i + 1, "embedded, %.2f s a page" % ((time.time() - t0) / (i + 1)), flush=True)
ok = [i for i, e in enumerate(E_) if e is not None]; data = data.iloc[ok].reset_index(drop=True); X = np.stack([E_[i] for i in ok]); thumbs = [thumbs[i] for i in ok]
X = X / np.linalg.norm(X, axis=1, keepdims=True); np.save(OUT / "embeddings.npy", X); data.to_json(OUT / "pages.json", orient="records")
km = KMeans(K, n_init=4, random_state=0).fit(X); data["cluster"] = km.labels_
def sheet(idx, out, title=None, cols=12):
    rows = (len(idx) + cols - 1) // cols; c = Image.new("RGB", (cols * 156, rows * 172 + (20 if title else 0)), "white"); dr = ImageDraw.Draw(c)
    if title: dr.text((3, 3), title, fill="black", font=FONT)
    for j, i in enumerate(idx):
        x, y = (j % cols) * 156, (j // cols) * 172 + (20 if title else 0); c.paste(thumbs[i], (x + 3, y + 16)); dr.text((x + 3, y + 2), f"{data.scan[i][:13]} L{data.lines[i]}", fill="black", font=FONT)
    c.save(out, quality=80)
order = np.argsort(-np.bincount(km.labels_, minlength=K)); summary = []
tiles = []
for rank, c in enumerate(order):
    members = np.where(km.labels_ == c)[0]; dist = np.linalg.norm(X[members] - km.cluster_centers_[c], axis=1); pick = members[np.argsort(dist)][:12]
    g = data.iloc[members]; summary.append({"cluster": int(c), "pages": len(members), "median_lines": float(g.lines.median()), "no_lines": float((g.lines == 0).mean()), "svea": float((g.pool == "svea").mean())})
    sheet(list(pick), OUT / f"cluster_{rank:02d}.jpg", f"cluster {rank}: {len(members)} pages, median lines {g.lines.median():.0f}, no-line share {(g.lines == 0).mean():.2f}, svea share {(g.pool == 'svea').mean():.2f}")
json.dump(summary, open(OUT / "clusters.json", "w")); print(pd.DataFrame(summary).round(2).to_string(), flush=True)
S_ = X @ X.T; np.fill_diagonal(S_, 0)
for name in known.scan.unique():
    i = data.index[data.scan == name]
    if len(i): nn = np.argsort(-S_[i[0]])[:11]; sheet([i[0]] + list(nn), OUT / f"near_{name}.jpg", f"nearest to {name} (first tile), cosine " + " ".join(f"{S_[i[0], j]:.2f}" for j in nn))
pairs = np.argwhere(np.triu(S_, 1) > 0.97); print("pairs with cosine > 0.97:", len(pairs), "> 0.985:", int((np.triu(S_, 1) > 0.985).sum()), flush=True)
best = sorted(pairs.tolist(), key=lambda p: -S_[p[0], p[1]])[:12]
if best: sheet([i for p in best for i in p], OUT / "near_duplicates.jpg", "most similar pairs (side by side): " + " ".join(f"{S_[a, b]:.3f}" for a, b in best), cols=12)
print("done", flush=True)
