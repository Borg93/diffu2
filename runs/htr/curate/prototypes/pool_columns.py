"""Columns for every scan from its patch vectors (CURATION E40): the classes marked by hand, table or running text,
kind, layout, and colour, each by a small head on the TIPSv2 and DINOv3 vectors.

    uv run python runs/htr/curate/prototypes/pool_columns.py fit      # the heads, from the 45,000 first described scans
    uv run python runs/htr/curate/prototypes/pool_columns.py fit_marks  # the cut class alone, from every hand mark (vectors from scans.lance)
    uv run python runs/htr/curate/prototypes/pool_columns.py fit_pca  # the PCA, from the first 200,000 scans of the pools' run
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
    pca                  the first 64 principal components of the 4,096 numbers (heads/pca.npz), 16-bit: for maps and
                         for nearest neighbours inside the table (scanner(nearest={"column": "pca", ...}, filter=...))

``apply`` must run alone on scans.lance (a new column is lost against a concurrent write).
"""
import os
import sys
import time
from pathlib import Path

# Before numpy and pyarrow load. With the machine's memory full of file cache, a fresh large array backed by huge pages
# stalls in the kernel (measured 18 Nov: a 154 MB part read in 61 s with the defaults, in 2 to 4 s with either of these).
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

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
PCA_FILE, PCA_DIMS, PCA_PARTS = HEADS / "pca.npz", 64, 10


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


KEPT: dict[str, dict[str, np.ndarray]] = {}  # the heads read so far


def scores(name: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A head's classes and its probability of each for the rows of ``X``."""
    if name not in KEPT:
        with np.load(HEADS / f"{name}.npz", allow_pickle=False) as held:
            KEPT[name] = {key: held[key] for key in ("classes", "coef", "intercept")}
    kept = KEPT[name]
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
    if PCA_FILE.exists():
        if "pca" not in KEPT:
            with np.load(PCA_FILE) as held:
                KEPT["pca"] = {"components": held["components"], "shift": held["mean"] @ held["components"].T}
        reduced = (X @ KEPT["pca"]["components"].T - KEPT["pca"]["shift"]).astype(np.float16)
        out["pca"] = pa.FixedSizeListArray.from_arrays(pa.array(reduced.ravel(), pa.float16()), PCA_DIMS)
    out["saturation"] = [None] * len(images) if saturation is None else saturation.astype(np.float32).tolist()
    out["colour"] = [None] * len(images) if saturation is None else (saturation > COLOUR).tolist()
    return out


def part_vectors(rows: pa.Table, X: np.ndarray) -> np.ndarray:
    """The heads' input for a part's rows, written into ``X``, which is allocated once: see the top of the file."""
    n = len(rows)
    for at, name in ((0, "tipsv2"), (2048, "dinov3")):
        half = X[:n, at:at + 2048]; row = 0
        for chunk in rows.column(name).chunks:  # the list column's values as they lie in the part, no array a row
            half[row:row + len(chunk)] = chunk.flatten().to_numpy(zero_copy_only=False).reshape(len(chunk), 2048); row += len(chunk)
        half /= np.linalg.norm(half, axis=1, keepdims=True)
    return X[:n]


SCHEMA = pa.schema([("image", pa.utf8()), ("cut_class", pa.utf8()), ("cut_p", pa.float32()), ("is_table", pa.bool_()), ("table_p", pa.float32()),
                    ("kind", pa.utf8()), ("kind_p", pa.float32()), ("layout", pa.utf8()), ("layout_p", pa.float32()),
                    *([("pca", pa.list_(pa.float16(), PCA_DIMS))] if PCA_FILE.exists() else []),
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
    elif what == "fit_marks":
        # Every marked scan's vectors from scans.lance (the gallery's marks reach beyond the 45,000 of embed_scans/):
        # each class with whole volumes held out, the new marks apart, then the head fitted on all and kept.
        mark = marks(); held = lance.dataset(str(HOME / "scans.lance"))
        got = pa.concat_tables([held.to_table(columns=["image", "tipsv2", "dinov3"], filter="image IN (" + ",".join("'" + i.replace("'", "''") + "'" for i in chunk) + ") AND tipsv2 IS NOT NULL")
                                for chunk in (list(mark.index[at:at + 2000]) for at in range(0, len(mark), 2000))])
        X = part_vectors(got, np.empty((len(got), 4096), np.float32)); images = np.array(got.column("image").to_pylist())
        first, first_X = sample(); more = (first.image.isin(mark.index) & ~first.image.isin(images)).to_numpy()  # marked scans the table holds no vectors of (no labelled lines)
        X = np.vstack([X, first_X[more]]); images = np.concatenate([images, first.image[more].to_numpy()]); y = mark.loc[images].to_numpy()
        counts = pd.Series(y).value_counts(); keep = np.isin(y, counts[counts >= 20].index); X, images, y = X[keep], images[keep], y[keep]
        print(f"{len(mark):,} marks; {len(y):,} with vectors in classes of 20 or more: {pd.Series(y).value_counts().to_dict()}", flush=True)
        volume = np.array([str(Path(image).parent) for image in images])
        guess = cross_val_predict(head(), X, y, groups=volume, cv=GroupKFold(5), n_jobs=5)
        print(classification_report(y, guess, digits=3, zero_division=0), flush=True)
        new = pd.read_parquet(HOME / "unsure/groups.parquet").image; fresh = np.isin(images, new)
        print(f"on the {int(fresh.sum())} scans marked on 20 Nov (the unsure ones), volumes held out: {np.mean(guess[fresh] == y[fresh]):.3f} right; on the others {np.mean(guess[~fresh] == y[~fresh]):.3f}", flush=True)
        print(pd.crosstab(pd.Series(y[fresh], name="mark"), pd.Series(guess[fresh], name="head")).to_string(), flush=True)
        fitted = head().fit(X, y)
        np.savez(HEADS / "cut_class.npz", classes=fitted.classes_.astype(str), coef=fitted.coef_.astype(np.float32), intercept=fitted.intercept_.astype(np.float32))
        print("kept in", HEADS / "cut_class.npz", fitted.coef_.shape, flush=True)
    elif what == "fit_pca":
        from sklearn.decomposition import PCA
        parts = sorted(PARTS[0].glob("part-*.parquet"))[:PCA_PARTS]; X = np.empty((20000 * len(parts), 4096), np.float32); n = 0
        for part in parts:  # the run's order is shuffled, so its first parts are a random sample of the pools
            rows = pq.read_table(part, columns=["tipsv2", "dinov3"]); part_vectors(rows, X[n:n + len(rows)]); n += len(rows)
        fitted = PCA(n_components=PCA_DIMS, svd_solver="randomized", random_state=0, copy=False).fit(X[:n])
        np.savez(PCA_FILE, mean=fitted.mean_.astype(np.float32), components=fitted.components_.astype(np.float32), explained=fitted.explained_variance_ratio_.astype(np.float32))
        share = fitted.explained_variance_ratio_.cumsum()
        print(f"{n:,} scans; the first 2, 8, 16, 32, 64 components hold {share[1]:.3f}, {share[7]:.3f}, {share[15]:.3f}, {share[31]:.3f}, {share[63]:.3f} of the variance; kept in {PCA_FILE}", flush=True)
    elif what == "apply":
        tables, seen = [], set(); start = time.time(); room = np.empty((20000, 4096), np.float32)
        folders = [Path(sys.argv[2])] if len(sys.argv) > 2 else PARTS
        parts = sorted(part for folder in folders for part in folder.glob("part-*.parquet"))
        for done, part in enumerate(parts, 1):
            rows = pq.read_table(part)
            X = part_vectors(rows, room if len(rows) <= len(room) else np.empty((len(rows), 4096), np.float32))
            images = rows.column("image").to_pylist(); fresh = np.array([image not in seen for image in images])
            seen.update(images); saturation = rows.column("saturation").to_numpy()
            if not fresh.all():  # a scan two parts hold keeps its first
                images, X, saturation = [image for image, keep in zip(images, fresh) if keep], X[fresh], saturation[fresh]
            tables.append(pa.table(columns(images, X, saturation), schema=SCHEMA))
            if done % 10 == 0 or done == len(parts):
                print(f"{done} of {len(parts)} parts, {len(seen):,} scans, {time.time() - start:.0f} s", flush=True)
        scans, X = sample(); fresh = ~scans.image.isin(seen).to_numpy()  # the embedded scans the runs do not hold
        if fresh.any() and len(sys.argv) <= 2:
            tables.append(pa.table(columns(scans.image[fresh].tolist(), X[fresh], None), schema=SCHEMA))
        features = pa.concat_tables(tables); held = lance.dataset(str(HOME / "scans.lance"))
        if "saturation" in held.schema.names:  # a scan without a saturation here keeps the one the table holds
            features = features.drop_columns(["saturation", "colour"]) if features.column("saturation").null_count == len(features) else features
        print(f"{len(features):,} scans with columns after {time.time() - start:.0f} s; joining", flush=True); joining = time.time()
        print(f"{add_features(HOME / 'scans.lance', features):,} matched in scans.lance; joined in {time.time() - joining:.0f} s", flush=True)
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
