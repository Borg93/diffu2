"""Which embedder maps the pages best? EVoC on each of the four page embeddings of the same 49,998 pages; on the
10,000 ground-truth pages among them, how well the clusters match what we know (collection, volume as a proxy for
series and hand) by V-measure and adjusted mutual information; noise share, cluster count, duplicates."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, evoc
from sklearn.metrics import v_measure_score, adjusted_mutual_info_score, homogeneity_score, completeness_score
E = Path(sys.argv[1]); pages = pd.read_json(E / "pages.json")
ok = np.load(E / "ok.npy") & np.load(E / "egemma_ok.npy")
models = {"dinov3": np.load(E / "dinov3.npy"), "tipsv2": np.load(E / "tipsv2.npy"), "egemma2_image": np.load(E / "egemma_image.npy"), "egemma2_image_text": np.load(E / "egemma_image_text.npy")}
gt = (pages.pool == "gt").values & ok
res = []
for name, X in models.items():
    X = X[ok].astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    c = evoc.EVoC(random_state=0); lab = np.full(len(pages), -2); lab[ok] = c.fit_predict(X)
    layers = getattr(c, "cluster_layers_", [lab[ok]])
    np.save(E / f"evoc_{name}.npy", lab)
    row = {"model": name, "clusters": int(lab.max()) + 1, "noise": round(float(np.mean(lab[ok] < 0)), 3), "layers": [int(l.max()) + 1 for l in layers], "dup_groups": len(getattr(c, "duplicates_", []) or [])}
    g = pages[gt]; l = lab[gt]
    for target in ("collection", "volume"):
        y = g[target].astype(str).values
        keep = l >= 0
        row[f"{target}_v"] = round(v_measure_score(y[keep], l[keep]), 3); row[f"{target}_ami"] = round(adjusted_mutual_info_score(y[keep], l[keep]), 3)
        row[f"{target}_homog"] = round(homogeneity_score(y[keep], l[keep]), 3); row[f"{target}_compl"] = round(completeness_score(y[keep], l[keep]), 3)
    # also on the finest layer, where hands should separate
    fl = np.full(len(pages), -2); fl[ok] = layers[0]; l = fl[gt]; keep = l >= 0; y = pages[gt].volume.astype(str).values
    row["volume_v_finest"] = round(v_measure_score(y[keep], l[keep]), 3)
    res.append(row); print(row, flush=True)
pd.DataFrame(res).to_json(E / "compare.json", orient="records"); print("done", flush=True)
