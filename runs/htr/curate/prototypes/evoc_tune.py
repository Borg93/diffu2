"""EVoC's settings tried for one embedding of the 15,000 described scans (CURATION E23).

Its defaults leave 40-70 % of these scans in no cluster. Varied: noise_level (0 clusters more data, 1 discards more),
n_neighbors of its graph, base_min_cluster_size (the finest layer's smallest cluster). Every layer of every setting is
scored on ALL scans, the unclustered counted as one more group, so a clustering cannot score well by leaving the hard
scans out:

- kind: adjusted mutual information with what the vision-language model says the scan is (five classes).
- volume: how pure the clusters are in the scan's volume (homogeneity): a volume is a series in a few hands.
- noise: the share of scans in no cluster.

    uv run --with evoc python runs/htr/curate/prototypes/evoc_tune.py NAME    (nvembed8b | qwen8b | egemma2 | tipsv2 | dinov3)
"""
import itertools, json, sys, time
from pathlib import Path
import evoc, numpy as np, pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score

E = Path("runs/htr/curate/embed_scans"); NAME = sys.argv[1]; rows = np.load(E / "sem_rows.npy")
sample = pd.read_json(E / "sample.json").iloc[rows].reset_index(drop=True)
said = pd.read_json("runs/htr/curate/vl/sample.jsonl", lines=True); said = said[said.kind.notna()].drop_duplicates("image")
again = pd.read_json("runs/htr/curate/vl/sample_pass2.jsonl", lines=True); again = again[again.writing_layout.notna()].drop_duplicates("image")
d = sample[["image", "volume"]].merge(said[["image", "kind", "layout"]], on="image", how="left").merge(again[["image", "writing_layout"]], on="image", how="left")
def what(r) -> str | None:
    if pd.isna(r.kind) or pd.isna(r.writing_layout): return None
    if r.layout == "not_a_page" or r.kind in ("sign_or_title_card", "cover_or_spine", "blank", "printed_or_typed_text", "drawing_or_map"): return "not handwriting"
    if r.layout in ("single_page", "several_loose_sheets"): return "single page"
    if r.layout in ("opening_one_page_empty", "opening_both_pages_empty"): return "opening, a page empty"
    return "opening, table or ledger" if r.writing_layout == "table" else "opening, running text"
d["what"] = d.apply(what, axis=1)
if NAME in ("tipsv2", "dinov3"): X, ok = np.load(E / f"{NAME}.npy")[rows].astype(np.float32), np.load(E / "ok.npy")[rows]
else: X, ok = np.load(E / f"sem_{NAME}.npy").astype(np.float32), np.load(E / f"sem_{NAME}_ok.npy")
use = ok & d.what.notna().to_numpy(); X = X[use]; X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
kind, volume = d.what.to_numpy()[use], d.volume.astype(str).to_numpy()[use]; out = []
for noise, neighbours, smallest in itertools.product((0.0, 0.25, 0.5), (15, 40), (5, 25)):
    started = time.time(); model = evoc.EVoC(noise_level=noise, n_neighbors=neighbours, base_min_cluster_size=smallest, random_state=0); model.fit(X)
    for depth, layer in enumerate(model.cluster_layers_):
        groups = np.where(layer < 0, layer.max() + 1, layer)  # the unclustered as one more group
        out.append({"noise_level": noise, "n_neighbors": neighbours, "base_min_cluster_size": smallest, "layer": depth, "clusters": int(layer.max()) + 1,
                    "noise": round(float((layer < 0).mean()), 3), "kind_ami": round(float(adjusted_mutual_info_score(kind, groups)), 3), "volume_homogeneity": round(float(homogeneity_score(volume, groups)), 3)})
    print(f"{NAME} noise_level {noise} n_neighbors {neighbours} smallest {smallest}: {[(o['clusters'], o['noise'], o['kind_ami']) for o in out if (o['noise_level'], o['n_neighbors'], o['base_min_cluster_size']) == (noise, neighbours, smallest)]} in {time.time() - started:.0f} s", flush=True)
    json.dump(out, open(E / f"evoc_tune_{NAME}.json", "w"), indent=1)
print("done", flush=True)
