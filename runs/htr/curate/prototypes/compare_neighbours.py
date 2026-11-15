"""Which embedder puts a page among its own: of each ground-truth page's 10 nearest ground-truth pages (cosine), the
share from its own volume and from its own collection, for every embedding of the comparison's pages found in the
folder. Independent of any clustering.

    uv run python runs/htr/curate/prototypes/compare_neighbours.py runs/htr/curate/embed
"""
import sys
from pathlib import Path
import lance, numpy as np, pandas as pd

E = Path(sys.argv[1]); pages = pd.read_json(E / "pages.json")
# the sample was drawn before the page table's collections were put right (3417d95): take them from the table
truth = lance.dataset("runs/htr/curate/pages.lance").to_table(columns=["page", "collection"], filter="pool = 'gt'").to_pandas().drop_duplicates("page").set_index("page").collection
pages["collection"] = pages.page.map(truth)
found = {"tipsv2": ("tipsv2.npy", "ok.npy"), "dinov3": ("dinov3.npy", "ok.npy"), "egemma2 image": ("egemma_image.npy", "egemma_ok.npy"), "egemma2 image+text": ("egemma_image_text.npy", "egemma_ok.npy"),
         "qwen3-vl-emb 2B image": ("qwen2b_image.npy", "qwen2b_ok.npy"), "qwen3-vl-emb 2B image+text": ("qwen2b_image_text.npy", "qwen2b_ok.npy"),
         "qwen3-vl-emb 8B image": ("qwen8b_image.npy", "qwen8b_ok.npy"), "qwen3-vl-emb 8B image+text": ("qwen8b_image_text.npy", "qwen8b_ok.npy")}
found = {name: files for name, files in found.items() if (E / files[0]).exists()}
ok = np.logical_and.reduce([np.load(E / okay) for _, okay in found.values()]); gt = ((pages.pool == "gt").to_numpy() & ok)
volume, collection = pages.volume.astype(str).to_numpy()[gt], pages.collection.astype(str).to_numpy()[gt]
print(int(gt.sum()), "ground-truth pages,", len(set(volume)), "volumes,", len(set(collection)), "collections")
for name, (vectors, _) in found.items():
    X = np.load(E / vectors)[gt].astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    near = np.empty((len(X), 10), int)
    for start in range(0, len(X), 2000):
        sim = X[start:start + 2000] @ X.T; sim[np.arange(len(sim)), np.arange(start, start + len(sim))] = -2
        near[start:start + 2000] = np.argpartition(-sim, 10, axis=1)[:, :10]
    print(f"{name:28s} same volume {(volume[near] == volume[:, None]).mean():.3f}   same collection {(collection[near] == collection[:, None]).mean():.3f}   ({X.shape[1]}-d)")
