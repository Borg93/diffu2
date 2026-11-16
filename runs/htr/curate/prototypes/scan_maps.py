"""Clusters and maps of the described scans from each one-vector-a-scan embedder (CURATION E21).

For every embedding of the 15,000 scans found in runs/htr/curate/embed_scans (sem_<name>.npy; the whole-scan TIPSv2
and DINOv3 vectors only with WITH_PATCH_MODELS=1: their job is where the writing is, not clustering): EVoC's layers of clusters, a 2-D UMAP,
and how well each tells apart what the scans are by the vision-language model's answers:

- neighbours: of a scan's 10 nearest, the share of its own class, averaged over the classes (so the common class
  does not carry it); no clustering involved.
- clusters: adjusted mutual information between an EVoC layer and the classes, and the share of scans EVoC leaves
  out as noise.

The classes: not handwriting; single page; opening with a page or both empty; opening written on both pages, running
text; the same, table or ledger (table against running text from the second pass).

    uv run --with evoc --with umap-learn --with matplotlib python runs/htr/curate/prototypes/scan_maps.py
"""
import json
from pathlib import Path
import evoc, numpy as np, pandas as pd, umap
from sklearn.metrics import adjusted_mutual_info_score

E = Path("runs/htr/curate/embed_scans"); rows = np.load(E / "sem_rows.npy"); sample = pd.read_json(E / "sample.json").iloc[rows].reset_index(drop=True)
said = pd.read_json("runs/htr/curate/vl/sample.jsonl", lines=True); said = said[said.kind.notna()].drop_duplicates("image")
again = pd.read_json("runs/htr/curate/vl/sample_pass2.jsonl", lines=True); again = again[again.writing_layout.notna()].drop_duplicates("image")
d = sample[["image", "pool", "volume"]].merge(said[["image", "kind", "layout", "caption"]], on="image", how="left").merge(again[["image", "writing_layout", "turned"]], on="image", how="left")
CLASSES = ["opening, running text", "opening, table or ledger", "opening, a page empty", "single page", "not handwriting"]
def what(r) -> str | None:
    if pd.isna(r.kind) or pd.isna(r.writing_layout): return None
    if r.layout == "not_a_page" or r.kind in ("sign_or_title_card", "cover_or_spine", "blank", "printed_or_typed_text", "drawing_or_map"): return CLASSES[4]
    if r.layout in ("single_page", "several_loose_sheets"): return CLASSES[3]
    if r.layout in ("opening_one_page_empty", "opening_both_pages_empty"): return CLASSES[2]
    return CLASSES[1] if r.writing_layout == "table" else CLASSES[0]
d["what"] = d.apply(what, axis=1); known = d.what.notna().to_numpy()
print(len(d), "scans,", int(known.sum()), "with both passes' answers;", d.what.value_counts().to_dict())
found = {name: (E / f"sem_{name}.npy", E / f"sem_{name}_ok.npy") for name in ("nvembed8b", "qwen8b", "egemma2") if (E / f"sem_{name}.npy").exists()}
vectors = {name: (np.load(v).astype(np.float32), np.load(o)) for name, (v, o) in found.items()}
whole_ok = np.load(E / "ok.npy")[rows]
import os
if os.environ.get("WITH_PATCH_MODELS"):  # off by default: DINOv3 and TIPSv2 have another job (where the writing is), not this one
    for name in ("tipsv2", "dinov3"): vectors[name + " (reference)"] = (np.load(E / f"{name}.npy")[rows].astype(np.float32), whole_ok)
results, maps = [], {}
for name, (X, ok) in vectors.items():
    use = ok & known; X = X[use]; X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8; y = d.what.to_numpy()[use]
    near = np.empty((len(X), 10), int)
    for start in range(0, len(X), 2000):
        sim = X[start:start + 2000] @ X.T; sim[np.arange(len(sim)), np.arange(start, start + len(sim))] = -2; near[start:start + 2000] = np.argpartition(-sim, 10, axis=1)[:, :10]
    same = (y[near] == y[:, None]).mean(axis=1); by_class = {c: float(same[y == c].mean()) for c in CLASSES}
    model = evoc.EVoC(random_state=0); model.fit_predict(X); layers = []
    for layer in model.cluster_layers_:
        inside = layer >= 0; layers.append({"clusters": int(layer.max()) + 1, "noise": round(float((~inside).mean()), 3), "ami": round(float(adjusted_mutual_info_score(y[inside], layer[inside])), 3)})
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X)
    middle = min(range(len(layers)), key=lambda i: abs(layers[i]["clusters"] - 60)); np.save(E / f"map_{name.split()[0]}.npy", xy)
    maps[name] = {"xy": xy, "y": y, "cluster": model.cluster_layers_[middle], "images": d.image.to_numpy()[use]}
    row = {"embedder": name, "scans": int(use.sum()), "neighbours_same_class_balanced": round(float(np.mean(list(by_class.values()))), 3), **{f"nn {c}": round(v, 3) for c, v in by_class.items()}, "layers": layers}
    results.append(row); print(json.dumps(row), flush=True)
chance = float(np.mean([(d.what[known] == c).mean() for c in CLASSES])); print("chance for the balanced neighbour score: %.3f" % chance)
json.dump(results, open(E / "maps.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
COLOUR = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))  # the dataviz palette's first five slots, validated
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a63", "#fcfcfb"
TITLE = {"nvembed8b": "Nemotron 3.5 Embed 8B", "qwen8b": "Qwen3-VL-Embedding 8B", "egemma2": "EmbeddingGemma 2", "tipsv2 (reference)": "TIPSv2, whole-scan vector (reference)", "dinov3 (reference)": "DINOv3, whole-scan vector (reference)"}
fig, axes = plt.subplots(1, len(maps), figsize=(5.2 * len(maps), 5.6), facecolor=SURFACE)
for ax, (name, m) in zip(np.atleast_1d(axes), maps.items()):
    order = np.random.default_rng(0).permutation(len(m["xy"]))
    ax.scatter(m["xy"][order, 0], m["xy"][order, 1], c=[COLOUR[c] for c in m["y"][order]], s=3, linewidths=0, alpha=0.75)
    score = next(r for r in results if r["embedder"] == name)
    ax.set_title(f"{TITLE[name]}\nneighbours of the same kind: {score['neighbours_same_class_balanced']:.2f}", fontsize=11, color=INK, loc="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
    for side in ax.spines.values(): side.set_color("#dddcd6")
handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=COLOUR[c], label=c) for c in CLASSES]
fig.legend(handles=handles, loc="lower center", ncol=len(CLASSES), frameon=False, fontsize=10, labelcolor=INK)
fig.suptitle("15,000 scans placed by each embedder, coloured by what the vision-language model says the scan is", fontsize=13, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 0.94)); fig.savefig(E / "maps.png", dpi=110, facecolor=SURFACE); print("maps.png written")
