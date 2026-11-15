"""EVoC on the DINOv3 page embeddings already made (page_embed.py): its layers of clusters, its duplicates, and a 2-D
map (UMAP) clustered with fast_hdbscan, each cluster shown as a sheet of its pages."""
import sys, json, collections
from pathlib import Path
import numpy as np, pandas as pd, evoc, fast_hdbscan, umap
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
src, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
X = np.load(src / "embeddings.npy").astype(np.float32); pages = pd.read_json(src / "pages.json")
print("pages", len(X), "dims", X.shape[1], flush=True)
clusterer = evoc.EVoC(random_state=0)
labels = clusterer.fit_predict(X)
layers = getattr(clusterer, "cluster_layers_", [labels])
print("EVoC layers:", [int(l.max()) + 1 for l in layers], "| finest layer noise share %.2f" % np.mean(layers[0] < 0), "| chosen layer clusters", int(labels.max()) + 1, "noise %.2f" % np.mean(labels < 0), flush=True)
dups = getattr(clusterer, "duplicates_", None)
if dups is not None: print("EVoC duplicate groups:", len(dups), "pages in them:", sum(len(d) for d in dups), flush=True)
xy = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=0).fit_transform(X)
hl = fast_hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(xy)
print("fast_hdbscan on the 2-D map: clusters", int(hl.max()) + 1, "noise %.2f" % np.mean(hl < 0), flush=True)
pages["evoc"] = labels; pages["map_x"], pages["map_y"] = xy[:, 0], xy[:, 1]; pages["hdbscan"] = hl
for i, l in enumerate(layers): pages[f"evoc_layer{i}"] = l
pages.to_json(OUT / "pages_clustered.json", orient="records")
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
fig, ax = plt.subplots(figsize=(11, 9)); ax.scatter(xy[:, 0], xy[:, 1], c=np.where(labels < 0, -1, labels), s=4, cmap="tab20", alpha=0.8)
for c in range(int(labels.max()) + 1):
    m = labels == c; ax.text(xy[m, 0].mean(), xy[m, 1].mean(), str(c), fontsize=9, weight="bold")
ax.set_title("DINOv3 page embeddings, UMAP; colours and numbers: EVoC clusters"); ax.set_xticks([]); ax.set_yticks([]); fig.savefig(OUT / "map.png", dpi=110, bbox_inches="tight")
def thumb(r, size=140):
    with Image.open(r["image"]) as im:
        im.draft("RGB", (900, 900)); k = im.width / r["scan_width"]
        p = im.convert("RGB").crop((round(r["x0"] * k), round(r["y0"] * k), round(r["x1"] * k), round(r["y1"] * k)))
    p.thumbnail((size, size)); return p
summary = []
for c in sorted(set(labels.tolist())):
    g = pages[pages.evoc == c]; pick = g.sample(min(12, len(g)), random_state=0)
    summary.append({"cluster": int(c), "pages": len(g), "median_lines": float(g.lines.median()), "no_lines": round(float((g.lines == 0).mean()), 2), "svea": round(float((g.pool == "svea").mean()), 2)})
    sheet = Image.new("RGB", (12 * 146, 172), "white"); d = ImageDraw.Draw(sheet)
    d.text((3, 2), f"EVoC cluster {c}: {len(g)} pages, median lines {g.lines.median():.0f}, no-line {(g.lines == 0).mean():.2f}, svea {(g.pool == 'svea').mean():.2f}", fill="black", font=F)
    for j, r in enumerate(pick.to_dict("records")):
        try: sheet.paste(thumb(r), (j * 146 + 3, 22))
        except Exception: pass
    sheet.save(OUT / f"evoc_{'noise' if c < 0 else f'{c:02d}'}.jpg", quality=80)
print(pd.DataFrame(summary).to_string(), flush=True)
print("done", flush=True)
