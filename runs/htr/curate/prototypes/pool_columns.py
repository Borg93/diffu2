"""Columns for every scan from its patch vectors (CURATION E40): the classes marked by hand, table or running text,
kind, layout, and colour, each by a small head on the TIPSv2 and DINOv3 vectors.

    uv run python runs/htr/curate/prototypes/pool_columns.py fit      # the heads, from the 45,000 first described scans
    uv run python runs/htr/curate/prototypes/pool_columns.py check    # each head with whole volumes held out
    uv run python runs/htr/curate/prototypes/pool_columns.py apply [FOLDER]  # the heads over every embedded scan (or one folder of parts), joined to scans.lance
    uv run python runs/htr/curate/prototypes/pool_columns.py report   # what the columns say of the pools

A head is a logistic regression on the two models' normalised vectors side by side (4,096 numbers), balanced by
class, as E39 and E41 measured it. The heads are fitted on the scans of embed_scans/ (their vectors made the same
way as the pools': embed_pool.py reproduces them to a cosine of 0.9999) and kept as plain arrays in heads/*.npz.

    cut_class, cut_p     the class of the hand marks (cut, no_cut, single, not_a_page, odd) and P(cut)
    is_table, table_p    table or ledger (the VL model's defined answer) and its probability
    kind, kind_p         the VL model's kind of scan and the head's probability for it
    layout, layout_p     the VL model's layout and the head's probability for it
    saturation, colour   the scan's mean saturation and whether it is a colour scan (above 0.05)

``apply`` must run alone on scans.lance (a new column is lost against a concurrent write).
"""
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import normalize

from cut_head import marks
from diffu_curate.table import add_features

HOME = Path("runs/htr/curate"); E = HOME / "embed_scans"; HEADS = HOME / "heads"
PARTS = [Path("/mnt/scratch/gabriel/htr/embed_pool"), Path("/mnt/scratch/gabriel/htr/embed_gt")]  # embed_pool.py's outputs: the pools, the ground truth
COLOUR = 0.05  # a scan's mean saturation from which it is a colour scan (greyscale microfilm lies at 0)


def sample() -> tuple[pd.DataFrame, np.ndarray]:
    """The scans of embed_scans/ that were embedded, with what is known of each, and their vectors."""
    scans = pd.read_json(E / "sample.json")[["image"]]; ok = np.load(E / "ok.npy")
    X = np.hstack([normalize(np.load(E / "tipsv2.npy")[ok].astype(np.float32)), normalize(np.load(E / "dinov3.npy")[ok].astype(np.float32))])
    scans = scans[ok].reset_index(drop=True)
    said = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", "vl_kind", "vl_layout", "vl_writing_layout"], filter="vl_kind IS NOT NULL").to_pandas()
    scans = scans.merge(said.drop_duplicates("image"), on="image", how="left")
    scans["cut_class"] = scans.image.map(marks())
    scans["table"] = scans.vl_writing_layout.map({"table": "table", "running_text": "running_text"})
    scans["volume"] = scans.image.map(lambda image: str(Path(image).parent))
    return scans, X


TARGETS = {"cut_class": "cut_class", "table": "table", "kind": "vl_kind", "layout": "vl_layout"}  # head: the column it learns


def labelled(scans: pd.DataFrame, column: str) -> np.ndarray:
    """The rows that hold a label of ``column`` whose class has 20 or more scans."""
    counts = scans[column].value_counts()
    return scans[column].isin(counts[counts >= 20].index).to_numpy()


def head() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, C=10, class_weight="balanced")


def scores(name: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A head's classes and its probability of each for the rows of ``X``."""
    kept = np.load(HEADS / f"{name}.npz", allow_pickle=False)
    logits = X @ kept["coef"].T + kept["intercept"]
    if logits.shape[1] == 1:  # two classes: one row of weights, for the second
        p = 1 / (1 + np.exp(-logits[:, 0])); return kept["classes"], np.stack([1 - p, p], axis=1)
    logits -= logits.max(axis=1, keepdims=True); p = np.exp(logits)
    return kept["classes"], p / p.sum(axis=1, keepdims=True)


def columns(images: list[str], X: np.ndarray, saturation: np.ndarray | None) -> dict[str, list]:
    out: dict[str, list] = {"image": images}
    for name in TARGETS:
        classes, p = scores(name, X); best = p.argmax(axis=1)
        if name == "cut_class":
            out["cut_class"] = classes[best].tolist(); out["cut_p"] = p[:, list(classes).index("cut")].astype(np.float32).tolist()
        elif name == "table":
            table = p[:, list(classes).index("table")]; out["is_table"] = (table >= 0.5).tolist(); out["table_p"] = table.astype(np.float32).tolist()
        else:
            out[name] = classes[best].tolist(); out[f"{name}_p"] = p[np.arange(len(best)), best].astype(np.float32).tolist()
    out["saturation"] = [None] * len(images) if saturation is None else saturation.astype(np.float32).tolist()
    out["colour"] = [None] * len(images) if saturation is None else (saturation > COLOUR).tolist()
    return out


SCHEMA = pa.schema([("image", pa.utf8()), ("cut_class", pa.utf8()), ("cut_p", pa.float32()), ("is_table", pa.bool_()), ("table_p", pa.float32()),
                    ("kind", pa.utf8()), ("kind_p", pa.float32()), ("layout", pa.utf8()), ("layout_p", pa.float32()),
                    ("saturation", pa.float32()), ("colour", pa.bool_())])

if __name__ == "__main__":
    what = sys.argv[1]
    if what in ("fit", "check"):
        scans, X = sample(); HEADS.mkdir(exist_ok=True)
        for name, column in TARGETS.items():
            rows = labelled(scans, column); y = scans[column][rows].to_numpy()
            print(f"\n== {name}: {rows.sum()} scans; {pd.Series(y).value_counts().to_dict()}", flush=True)
            if what == "check":
                guess = cross_val_predict(head(), X[rows], y, groups=scans.volume[rows].to_numpy(), cv=GroupKFold(5), n_jobs=5)
                print(classification_report(y, guess, digits=3, zero_division=0), flush=True)
            else:
                fitted = head().fit(X[rows], y)
                np.savez(HEADS / f"{name}.npz", classes=fitted.classes_.astype(str), coef=fitted.coef_.astype(np.float32), intercept=fitted.intercept_.astype(np.float32))
                print("kept in", HEADS / f"{name}.npz", fitted.coef_.shape, flush=True)
    elif what == "apply":
        tables, seen = [], set()
        folders = [Path(sys.argv[2])] if len(sys.argv) > 2 else PARTS
        for part in sorted(part for folder in folders for part in folder.glob("part-*.parquet")):
            rows = pq.read_table(part)
            X = np.hstack([normalize(np.stack(rows.column("tipsv2").to_numpy(zero_copy_only=False)).astype(np.float32)),
                           normalize(np.stack(rows.column("dinov3").to_numpy(zero_copy_only=False)).astype(np.float32))])
            images = rows.column("image").to_pylist(); fresh = np.array([image not in seen for image in images])
            seen.update(images)
            got = columns([image for image, keep in zip(images, fresh) if keep], X[fresh], rows.column("saturation").to_numpy()[fresh])
            tables.append(pa.table(got, schema=SCHEMA)); print(part.name, len(images), flush=True)
        scans, X = sample(); fresh = ~scans.image.isin(seen).to_numpy()  # the embedded scans the runs do not hold
        if fresh.any() and len(sys.argv) <= 2:
            tables.append(pa.table(columns(scans.image[fresh].tolist(), X[fresh], None), schema=SCHEMA))
        features = pa.concat_tables(tables); held = lance.dataset(str(HOME / "scans.lance"))
        if "saturation" in held.schema.names:  # a scan without a saturation here keeps the one the table holds
            features = features.drop_columns(["saturation", "colour"]) if features.column("saturation").null_count == len(features) else features
        print(len(features), "scans with columns;", add_features(HOME / "scans.lance", features), "matched in scans.lance", flush=True)
    elif what == "report":
        held = lance.dataset(str(HOME / "scans.lance")); pool = "pool != 'gt' AND cut_class IS NOT NULL"
        total = held.count_rows("pool != 'gt'"); done = held.count_rows(pool)
        print(f"{done:,} of {total:,} pool scans hold the columns")
        for name, where in (("the head says cut", "cut_class = 'cut'"), ("... and lines are labelled", "cut_class = 'cut' AND lines > 0"),
                            ("... and a fold is placed", "cut_class = 'cut' AND lines > 0 AND fold_found IS NOT NULL"),
                            ("... and a fold line is found", "cut_class = 'cut' AND lines > 0 AND fold_found = true"),
                            ("    of those, colour scans", "cut_class = 'cut' AND lines > 0 AND fold_found = true AND colour = true"),
                            ("    of those, tables", "cut_class = 'cut' AND lines > 0 AND fold_found = true AND is_table = true"),
                            ("single pages by the head", "cut_class = 'single'"), ("not a page by the head", "cut_class = 'not_a_page'"),
                            ("do not cut by the head", "cut_class = 'no_cut'"), ("odd layout by the head", "cut_class = 'odd'"),
                            ("tables by the head", "is_table = true"), ("colour scans", "colour = true")):
            count = held.count_rows(f"{pool} AND {where}"); print(f"{name:34} {count:10,}  {count / max(done, 1):6.1%}")
