"""A head on the scan embeddings that says whether a scan is to be cut, fitted on the marks made by hand (CURATION E36).

Marks: fold_marks (one scan at a time; a clicked fold is "cut"), then cut_marks (gallery) and region_marks (map),
the latest row of an image winning within a table and the fold marks over the others. Logistic head on the
normalised Qwen3-VL-Embedding 8B vectors; scored with the volumes held out, then fitted on every mark and run over
every scan in scan_embeddings.lance. Its answer and P(cut) become the columns head_cut and head_p_cut of scans.lance.

    uv run python runs/htr/curate/prototypes/cut_head.py
"""
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

from diffu_curate.table import add_features

HOME = Path("runs/htr/curate")


def latest(name: str) -> pd.Series:
    rows = lance.dataset(str(HOME / name)).to_table(columns=["image", "mark", "marked_at"]).to_pandas()
    return rows.sort_values("marked_at").drop_duplicates("image", keep="last").set_index("image").mark


def marks() -> pd.Series:
    bulk = pd.concat([latest("cut_marks.lance"), latest("region_marks.lance")])
    bulk = bulk[~bulk.index.duplicated(keep="last")]
    single = latest("fold_marks.lance").replace({"fold": "cut"}); single = single[single != "unsure"]
    return pd.concat([bulk[~bulk.index.isin(single.index)], single])


if __name__ == "__main__":
    held = lance.dataset(str(HOME / "scan_embeddings.lance")).to_table().to_pandas()
    X = normalize(np.stack(held.qwen8b.to_numpy()).astype(np.float32)); images = held.image.to_numpy()
    mark = marks(); known = np.isin(images, mark.index); y = mark.loc[images[known]].to_numpy()
    volume = np.array([str(Path(image).parent) for image in images[known]])
    print(known.sum(), "marked scans with a vector;", pd.Series(y).value_counts().to_dict(), flush=True)
    guess = np.empty(len(y), dtype=object); p_cut = np.zeros(len(y))
    for fit, out in GroupKFold(5).split(X[known], y, volume):
        head = LogisticRegression(max_iter=3000, C=10, class_weight="balanced").fit(X[known][fit], y[fit])
        guess[out] = head.predict(X[known][out]); p_cut[out] = head.predict_proba(X[known][out])[:, list(head.classes_).index("cut")]
    print(f"volumes held out: accuracy {np.mean(guess == y):.3f}"); print(pd.crosstab(pd.Series(y, name="mark"), pd.Series(guess, name="head")))
    not_cut = y != "cut"
    for floor in (None, 0.8, 0.9, 0.95):
        says = guess != "cut" if floor is None else p_cut < floor
        print(f"{'head says not cut' if floor is None else f'P(cut) < {floor}':18} not-to-cut found {says[not_cut].mean():.3f} | cut held back {says[~not_cut].mean():.3f}")
    head = LogisticRegression(max_iter=3000, C=10, class_weight="balanced").fit(X[known], y)
    answer = head.predict(X); p = head.predict_proba(X)[:, list(head.classes_).index("cut")]
    matched = add_features(HOME / "scans.lance", pa.table({"image": images, "head_cut": answer.astype(str), "head_p_cut": p.astype(np.float32)}))
    print(matched, "scans given head_cut;", pd.Series(answer).value_counts().to_dict())
