"""SCALAR and HiRank on the described scans (CURATION E25).

EVoC alone leaves 40-70 % of these scans in no cluster. Tried for full coverage, each asked for about 50 clusters:
  a. EVoC on every scan (the baseline).
  b. the same, its unclustered scans then voted into a cluster by their nearest clustered neighbours (SCALAR's
     KNNUpsampler used alone).
  c. SCALAR's pipeline: k-means representatives of the scans, EVoC on the representatives, labels spread back to every
     scan by nearest neighbours. This is also the form that scales to the 3.15 million scans.
Scored on all scans: the share left unclustered, and the adjusted mutual information with what the vision-language
model says the scan is (the unclustered counted as one more group).

HiRank's RankOD gives every scan an outlier score from reverse nearest-neighbour ranks. Scored by whether the rare
kinds get high scores (AUC against the common kind), and the 18 oddest scans are drawn to look at.

    uv run --with evoc --with git+https://github.com/TutteInstitute/scalar --with git+https://github.com/TutteInstitute/hirank \\
        python runs/htr/curate/prototypes/scalar_hirank.py NAME [SHEET.jpg]
"""
import sys
from pathlib import Path
import evoc, numpy as np, pandas as pd
from hirank import RankOD
from scalar import ClusterScalar
from scalar.downsampling import KMeansDownsampler
from scalar.upsampling import KNNUpsampler
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score

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
X, ok = np.load(E / f"sem_{NAME}.npy").astype(np.float32), np.load(E / f"sem_{NAME}_ok.npy")
use = ok & d.what.notna().to_numpy(); X = X[use]; X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8; d = d[use].reset_index(drop=True); kind = d.what.to_numpy()

def score(name: str, labels: np.ndarray) -> None:
    groups = np.where(labels < 0, labels.max() + 1, labels)
    sizes = pd.Series(labels[labels >= 0]).value_counts()
    print(f"{NAME} {name:58s} clusters {labels.max() + 1:4d}  unclustered {np.mean(labels < 0):.3f}  kind AMI {adjusted_mutual_info_score(kind, groups):.3f}  largest cluster {sizes.iloc[0] / len(labels):.2f}", flush=True)

import os
CLUSTER = not os.environ.get("ODD_ONLY")
alone = evoc.EVoC(approx_n_clusters=50, random_state=0).fit_predict(X) if CLUSTER else np.zeros(len(X), int); score("a. EVoC alone", alone) if CLUSTER else None
for weight in (0.1, 0.0) if CLUSTER else ():
    inside = alone >= 0
    voted = KNNUpsampler(n_neighbors=5, metric="cosine", noise_voting_weight=weight).upsample(X, X[inside], alone[inside], {}); voted[inside] = alone[inside]
    score(f"b. EVoC, the unclustered voted in (5 neighbours)", voted) if weight == 0.1 else None
if CLUSTER: np.save(E / f"evoc50_voted_{NAME}.npy", voted)
for centres in (1500, 4000) if CLUSTER else ():
    for weight in (0.1, 0.0):
        piped = ClusterScalar(clusterer=evoc.EVoC(approx_n_clusters=50, random_state=0), downsampler=KMeansDownsampler(n_clusters=centres, random_state=0),
                              upsampler=KNNUpsampler(n_neighbors=5, noise_voting_weight=weight)).fit(X).labels_
        score(f"c. SCALAR: {centres} k-means representatives, noise vote {weight}", np.asarray(piped))

# score_samples is a density: 0 for a scan in none of its neighbours' nearest-100 lists. HiRank's README says higher is
# more of an outlier, its docstring and the numbers say lower; checked on the scores themselves, so negated here.
odd = -RankOD(n_neighbors=15, max_rank=100, metric="cosine", dtype=np.float32).fit(X).score_samples(X)
np.save(E / f"hirank_{NAME}.npy", odd); common = kind == "opening, running text"
for rare in ("not handwriting", "single page", "opening, table or ledger", "opening, a page empty"):
    pick = common | (kind == rare); print(f"{NAME} HiRank: {rare:26s} scores above running-text openings, AUC {roc_auc_score(kind[pick] == rare, odd[pick]):.3f}  (n={int((kind == rare).sum())})")
top = np.argsort(-odd)[:300]; print(f"{NAME} HiRank: the 300 oddest scans are", pd.Series(kind[top]).value_counts().to_dict(), "| all scans:", pd.Series(kind).value_counts(normalize=True).round(3).to_dict())
if len(sys.argv) > 2:
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None; sheet = Image.new("RGB", (6 * 430, 3 * 330), "white")
    for j, i in enumerate(np.argsort(-odd)[:18]):
        with Image.open(d.image[i]) as scan: scan.draft("RGB", (860, 640)); scan = scan.convert("RGB"); scan.thumbnail((424, 300))
        sheet.paste(scan, ((j % 6) * 430, (j // 6) * 330 + 26)); ImageDraw.Draw(sheet).text(((j % 6) * 430 + 2, (j // 6) * 330 + 2), f"{j} density {-odd[i]:.4f} | {kind[i]}", fill="black")
    sheet.save(sys.argv[2], quality=80)
