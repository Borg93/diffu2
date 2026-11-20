"""A cluster and its name for every embedded scan, as columns of scans.lance: `pool_cluster`, `pool_cluster_name`
(CURATION E65).

    flock runs/htr/curate/scans.lock uv run python runs/htr/curate/prototypes/scan_clusters.py [CLUSTERS]

K-means (64 by default) on the `pca` column (the first 64 components of the DINOv3 and TIPSv2 vectors, each scan
of unit length), over the pools and the ground truth together. A cluster's name says what its scans mostly are by
the columns (class, table, colour, language, pool) and the words the VL model's captions use more in it than
elsewhere. A first, plain clustering: every scan gets a cluster, none is left out as noise.
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0"); os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import re, sys
from collections import Counter
from pathlib import Path

import lance, numpy as np, pandas as pd, pyarrow as pa
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from diffu_curate.table import add_features

HOME = Path("runs/htr/curate"); K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
held = lance.dataset(str(HOME / "scans.lance"))
t = held.to_table(columns=["image", "pool", "pca", "cut_class", "is_table", "colour", "language", "vl_caption"], filter="pca IS NOT NULL").to_pandas()
Z = np.stack(t.pca.to_numpy()).astype(np.float32); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
fitted = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=20000, n_init=3, max_iter=200).fit(Z)
t["cluster"] = fitted.labels_.astype(np.int32); print(len(t), "scans in", K, "clusters; sizes from", int(t.cluster.value_counts().min()), "to", int(t.cluster.value_counts().max()), flush=True)
captions = t[t.vl_caption.notna()].groupby("cluster").vl_caption.apply(" ".join)
words = TfidfVectorizer(stop_words="english", token_pattern=r"[a-zA-Z]{4,}", max_df=0.6, sublinear_tf=True).fit(captions); scores = words.transform(captions); vocab = np.array(words.get_feature_names_out())
top = {cluster: ", ".join(vocab[np.asarray(scores[row].todense()).ravel().argsort()[::-1][:4]]) for row, cluster in enumerate(captions.index)}
def share(part, mask): return float(mask.mean()) if len(part) else 0.0
names = {}
for cluster, part in t.groupby("cluster"):
    said = [f"{part.cut_class.value_counts(normalize=True).index[0]} {part.cut_class.value_counts(normalize=True).iloc[0]:.0%}"]
    if share(part, part.is_table.fillna(False).astype(bool)) > 0.3: said.append(f"tables {share(part, part.is_table.fillna(False).astype(bool)):.0%}")
    if share(part, part.colour.fillna(False).astype(bool)) > 0.5: said.append("colour")
    other = part.language[part.language.notna() & (part.language != "swedish")]
    if len(other) / max(1, part.language.notna().sum()) > 0.15: said.append(f"{other.value_counts().index[0]} {len(other) / part.language.notna().sum():.0%}")
    pool = part.pool.value_counts(normalize=True); said.append(f"{pool.index[0]} {pool.iloc[0]:.0%}")
    names[cluster] = f"{cluster:02d}: " + "; ".join(said) + (f" | {top[cluster]}" if cluster in top else "")
t["name"] = t.cluster.map(names)
for cluster in t.cluster.value_counts().index[:64]: print(f"{int((t.cluster == cluster).sum()):8,}  {names[cluster]}")
features = pa.table({"image": pa.array(t.image, pa.utf8()), "pool_cluster": pa.array(t.cluster, pa.int32()), "pool_cluster_name": pa.array(t.name, pa.utf8())})
print(f"{add_features(HOME / 'scans.lance', features):,} scans given pool_cluster and pool_cluster_name")
np.save(HOME / "heads/pool_cluster_centres.npy", fitted.cluster_centers_.astype(np.float32))
