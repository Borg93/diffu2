"""Is the vision-language model needed on every scan (CURATION E41)? Two checks on the scans that hold patch vectors.

1. Cut or not, against the hand marks: the VL model's first form, its defined question, the head on the patch
   vectors (volumes held out), and the head with the VL model's two answers, each by how many of the scans not to be
   cut it finds, how many openings it holds back, and what share of the scans it would cut are not to be cut.
2. Table or running text: a head on the patch vectors fitted on the VL model's defined answer (E20), volumes held out,
   scored against that answer.

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=runs/htr/curate/prototypes uv run python runs/htr/curate/prototypes/vl_or_head.py
"""
from pathlib import Path

import lance
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import normalize

from cut_head import marks

HOME = Path("runs/htr/curate"); E = HOME / "embed_scans"


def line(name: str, says_not: np.ndarray, not_cut: np.ndarray) -> None:
    print(f"{name:44} not-to-cut found {says_not[not_cut].mean():.3f} | openings held back {says_not[~not_cut].mean():.3f} | "
          f"kept {int((~says_not).sum()):6} | of the kept, not to be cut {not_cut[~says_not].mean():.4f}", flush=True)


if __name__ == "__main__":
    sample = pd.read_json(E / "sample.json"); ok = np.load(E / "ok.npy")
    X = np.hstack([normalize(np.load(E / "tipsv2.npy").astype(np.float32)), normalize(np.load(E / "dinov3.npy").astype(np.float32))])
    said = lance.dataset(str(HOME / "scans.lance")).to_table(
        columns=["image", "pool", "vl_layout", "vl_table_across_fold", "vl_cut", "vl_writing_layout"], filter="vl_kind IS NOT NULL").to_pandas().set_index("image")
    sample = sample.assign(at=range(len(sample)))[ok & sample.image.isin(said.index).to_numpy()]
    sample = sample.join(said[["vl_layout", "vl_table_across_fold", "vl_cut", "vl_writing_layout"]], on="image")
    volume = sample.image.map(lambda image: str(Path(image).parent)).to_numpy()

    mark = marks(); marked = sample[sample.image.isin(mark.index)]
    y = mark.loc[marked.image].to_numpy(); not_cut = y != "cut"
    print(len(y), "marked scans with patch vectors;", pd.Series(y).value_counts().to_dict(), flush=True)
    head = LogisticRegression(max_iter=3000, C=10, class_weight="balanced")
    guess = cross_val_predict(head, X[marked["at"].to_numpy()], y, groups=marked.image.map(lambda image: str(Path(image).parent)).to_numpy(), cv=GroupKFold(5), n_jobs=5)
    form_not = ~marked.vl_layout.isin(["opening_both_pages_written", "opening_one_page_empty"]).to_numpy() | marked.vl_table_across_fold.fillna(False).astype(bool).to_numpy()
    asked_not = form_not | (marked.vl_cut != "cut").to_numpy()
    head_not = guess != "cut"
    line("VL model, first form", form_not, not_cut)
    line("VL model, first form and defined question", asked_not, not_cut)
    line("head on patch vectors", head_not, not_cut)
    line("head and the VL model's two answers", head_not | asked_not, not_cut)
    missed = not_cut & ~head_not
    print("of the", int(missed.sum()), "scans not to be cut that the head would cut, the VL model's answers stop", int((missed & asked_not).sum()),
          "| of the", int((~not_cut & ~head_not).sum()), "openings the head keeps, the VL model's answers hold back", int((~not_cut & ~head_not & asked_not).sum()), flush=True)

    known = sample[sample.vl_writing_layout.isin(["table", "running_text"])]
    kind = known.vl_writing_layout.to_numpy()
    print("\n", len(kind), "scans with the VL model's defined answer;", pd.Series(kind).value_counts().to_dict(), flush=True)
    head = LogisticRegression(max_iter=3000, C=10, class_weight="balanced")
    guess = cross_val_predict(head, X[known["at"].to_numpy()], kind, groups=known.image.map(lambda image: str(Path(image).parent)).to_numpy(), cv=GroupKFold(5), n_jobs=5)
    print(classification_report(kind, guess, digits=3), flush=True)
