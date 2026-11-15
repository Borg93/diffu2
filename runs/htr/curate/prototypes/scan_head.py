"""How well a head on the scan embeddings gives the vision-language model's answers, class by class (CURATION E16).

The 45,000 described scans, TIPSv2 and DINOv3 embeddings of the whole scan joined and reduced to 256 dimensions; a
logistic head for each answer, fitted on four fifths of the volumes and scored on the fifth, five times. For the
filter what counts is each rare class's recall and precision, not the overall share right.

    uv run python runs/htr/curate/prototypes/scan_head.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

E = Path("runs/htr/curate/embed_scans"); ok = np.load(E / "ok.npy")
said = pd.read_json("runs/htr/curate/vl/sample.jsonl", lines=True); said = said[said.kind.notna()]
place = {image: i for i, image in enumerate(pd.read_json(E / "sample.json").image)}
rows = said.image.map(place).to_numpy(); keep = ok[rows]; said, rows = said[keep].reset_index(drop=True), rows[keep]
X = np.hstack([normalize(np.load(E / "tipsv2.npy")[rows].astype(np.float32)), normalize(np.load(E / "dinov3.npy")[rows].astype(np.float32))])
X = PCA(256, random_state=0).fit_transform(X); groups = said.volume.astype(str)
print(len(said), "described scans with an embedding;", said.pool.value_counts().to_dict())
said["what"] = np.where(said.layout == "not_a_page", "not_a_page", np.where(said.kind.isin(["sign_or_title_card", "cover_or_spine", "blank", "printed_or_typed_text", "drawing_or_map"]), said.kind, "handwriting"))
said["pages"] = said.layout.replace({"opening_both_pages_empty": "opening_one_or_both_empty", "opening_one_page_empty": "opening_one_or_both_empty", "several_loose_sheets": "single_page"})
said["handwriting"] = np.where(said.what == "handwriting", "handwriting", "other")
import os
for answer in os.environ.get("ANSWERS", "handwriting,what,pages,written_side").split(","):
    y = said[answer].astype(str).to_numpy(); guess = np.empty(len(y), dtype=object)
    for fit, held in GroupKFold(5).split(X, y, groups):
        guess[held] = LogisticRegression(max_iter=3000, C=3, class_weight="balanced").fit(X[fit], y[fit]).predict(X[held])
    print(f"\n== {answer}: right {np.mean(guess == y):.3f} (always the commonest {pd.Series(y).value_counts(normalize=True).iloc[0]:.3f})")
    print(classification_report(y, guess.astype(str), digits=3, zero_division=0))
