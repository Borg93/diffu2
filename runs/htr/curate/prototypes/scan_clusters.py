"""The described scans' clusters and how far the embeddings carry the model's answers (CURATION task 9).

EVoC on the TIPSv2 and DINOv3 embeddings of the scans in runs/htr/curate/embed_scans (whole scans, 448 px); then, on
the scans the vision-language model has described so far (runs/htr/curate/vl/sample.jsonl): how pure each cluster is
in the model's kind and layout, and how well a linear head on the embeddings gives the model's answer for a scan it
was not fitted on (5 folds, split by volume).

    uv run --with evoc python runs/htr/curate/prototypes/scan_clusters.py
"""
import json
from pathlib import Path
import evoc, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

E = Path("runs/htr/curate/embed_scans"); pages = pd.read_json(E / "pages.json"); ok = np.load(E / "ok.npy")
emb = {"tipsv2": np.load(E / "tipsv2.npy"), "dinov3": np.load(E / "dinov3.npy")}
for name, X in emb.items():
    if not (E / f"evoc_{name}.npy").exists():
        model = evoc.EVoC(random_state=0); labels = np.full(len(pages), -2); labels[ok] = model.fit_predict(X[ok])
        np.save(E / f"evoc_{name}.npy", labels)
        print(name, "clusters", labels.max() + 1, "noise %.2f" % (labels[ok] == -1).mean(), "layers", [int(l.max()) + 1 for l in model.cluster_layers_], flush=True)
        for depth, layer in enumerate(model.cluster_layers_):
            full = np.full(len(pages), -2); full[ok] = layer; np.save(E / f"evoc_{name}_layer{depth}.npy", full)
said = pd.read_json("runs/htr/curate/vl/sample.jsonl", lines=True); said = said[said.kind.notna()] if "kind" in said else said.iloc[:0]
place = {image: i for i, image in enumerate(pd.read_json(E / "sample.json").image)}
said = said[said.image.isin(place)]; rows = said.image.map(place).to_numpy(); keep = ok[rows]; said, rows = said[keep], rows[keep]
print("described scans with an embedding:", len(said), said.pool.value_counts().to_dict())
for answer in ("kind", "layout", "written_side"):
    print(answer, said[answer].value_counts().to_dict())
for name, X in emb.items():
    labels = np.load(E / f"evoc_{name}.npy")[rows]; d = said.assign(cluster=labels); d = d[d.cluster >= 0]
    for answer in ("kind", "layout"):
        top = d.groupby("cluster")[answer].agg(lambda s: s.value_counts(normalize=True).iloc[0]); size = d.groupby("cluster").size()
        print(f"{name} clusters, {answer}: a scan's cluster's commonest answer is its own for %.3f of clustered scans (always saying the commonest: %.3f); clusters at least 0.9 pure hold %.2f of them" % ((top * size).sum() / size.sum(), d[answer].value_counts(normalize=True).iloc[0], size[top >= 0.9].sum() / size.sum()))
feats = {**{k: normalize(v[rows]) for k, v in emb.items()}, "both": np.hstack([normalize(emb["tipsv2"][rows]), normalize(emb["dinov3"][rows])])}
for answer in ("kind", "layout", "written_side", "writing_at_fold"):
    y = said[answer].astype(str).to_numpy(); line = [f"{answer}: always the commonest %.3f" % pd.Series(y).value_counts(normalize=True).iloc[0]]
    for name, X in feats.items():
        hit = np.zeros(len(y), bool)
        for fit, held in GroupKFold(5).split(X, y, said.volume.astype(str)):
            hit[held] = LogisticRegression(max_iter=2000, C=10).fit(X[fit], y[fit]).predict(X[held]) == y[held]
        line.append(f"{name} %.3f" % hit.mean())
    print(" | ".join(line), flush=True)
