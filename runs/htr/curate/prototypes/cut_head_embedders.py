"""Does a head on the patch models' whole-scan vectors answer cut-or-not as well as the head on Qwen 8B's (CURATION E39)?

The hand marks (cut_head.marks) on the scans that hold both kinds of vector: Qwen3-VL-Embedding 8B's
(scan_embeddings.lance) and the TIPSv2 and DINOv3 class-and-mean-patch vectors of the 45,000 first described scans
(embed_scans/). The same logistic head and the same held-out volumes for each set of vectors. If the patch models do
as well, every pool scan can be embedded with a model a twentieth the size of the 8B one.

    OMP_NUM_THREADS=4 uv run python runs/htr/curate/prototypes/cut_head_embedders.py
"""
from pathlib import Path

import lance
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import normalize

from cut_head import marks

HOME = Path("runs/htr/curate"); E = HOME / "embed_scans"

if __name__ == "__main__":
    mark = marks()
    sample = pd.read_json(E / "sample.json").image.tolist(); ok = np.load(E / "ok.npy")
    place = {image: i for i, image in enumerate(sample) if ok[i]}
    held = lance.dataset(str(HOME / "scan_embeddings.lance"))
    images = held.to_table(columns=["image"]).column("image").to_pylist()
    rows = [i for i, image in enumerate(images) if image in place and image in mark.index]
    qwen = held.take(rows, columns=["image", "qwen8b"]).to_pandas()
    at = [place[image] for image in qwen.image]; y = mark.loc[qwen.image].to_numpy()
    volume = np.array([str(Path(image).parent) for image in qwen.image])
    print(len(y), "marked scans with both kinds of vector;", pd.Series(y).value_counts().to_dict(), flush=True)
    tips, dino = normalize(np.load(E / "tipsv2.npy")[at].astype(np.float32)), normalize(np.load(E / "dinov3.npy")[at].astype(np.float32))
    sets = {"Qwen3-VL-Embedding 8B": normalize(np.stack(qwen.qwen8b.to_numpy()).astype(np.float32)),
            "TIPSv2 + DINOv3": np.hstack([tips, dino]), "DINOv3": dino, "TIPSv2": tips}
    not_cut = y != "cut"
    for name, X in sets.items():
        head = LogisticRegression(max_iter=3000, C=10, class_weight="balanced")
        guess = cross_val_predict(head, X, y, groups=volume, cv=GroupKFold(5), n_jobs=5)
        says = guess != "cut"
        print(f"{name:22} {X.shape[1]:5} dims | right {np.mean(guess == y):.3f} | not-to-cut found {says[not_cut].mean():.3f} | "
              f"cut held back {says[~not_cut].mean():.3f} | of the scans it would cut, not to be cut {not_cut[~says].mean():.4f}", flush=True)
