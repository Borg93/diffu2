"""The pool scans to train on, taken by the columns every scan holds, staged for the set builder (CURATION E40).

A scan is taken when the old pipeline labelled lines on it, the head on its patch vectors calls it an opening to cut
(cut_class) with running text (is_table false) in handwriting (kind: not print or type), the placer finds a fold line (folds.lance), and its volume holds no
ground-truth scan. With SINGLES=1 the scans the head calls a single page are taken too, uncut. Below OUT the scans are
staged as stage_trusted.py stages them (per volume an ``alto/`` folder of links to the label files and links to the
scans; ``OUT/cuts.json`` each scan's fold for ``--cuts``), with ``OUT/scans.parquet`` the scans taken and their columns.

    uv run python runs/htr/curate/prototypes/stage_pool.py count               # the funnel over the pools, nothing staged
    uv run python runs/htr/curate/prototypes/stage_pool.py OUT [SCANS [SEED]]  # SCANS: so many at random (default all)
    EXCLUDE=a/scans.parquet,b/scans.parquet ... OUT ...                        # without the scans of sets staged before

The tables are the other filter (is_table true): ``count`` prints how many there are, by what the head says of the cut.
"""
import os

# Before numpy and pyarrow load: with the machine's memory full of file cache, large arrays on huge pages stall in the
# kernel (CURATION E40, 18 Nov).
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import pandas as pd

from diffu_curate.sample import truth_volumes
from diffu_curate.settings import CurateSettings

HOME = Path("runs/htr/curate")
COLUMNS = ["image", "pool", "volume", "scan", "lines", "source", "cut_class", "cut_p", "is_table", "table_p", "kind", "layout", "colour"]
SINGLES = os.environ.get("SINGLES") == "1"
HANDWRITING = ("handwritten_text", "handwritten_table_or_ledger")  # kind: the VL model's first form over-calls tables, is_table decides those


def pools() -> tuple[pd.DataFrame, int]:
    """Every pool scan that holds the columns, with its fold where one is placed; and how many pool scans there are."""
    held = lance.dataset(str(HOME / "scans.lance"))
    scans = held.to_table(columns=COLUMNS, filter="pool != 'gt' AND cut_class IS NOT NULL").to_pandas()
    folds = lance.dataset(str(HOME / "folds.lance")).to_table(columns=["image", "fold_x", "fold_found"]).to_pandas().drop_duplicates("image", keep="last")
    scans = scans.merge(folds, on="image", how="left")
    settings = CurateSettings()
    scans["in_truth"] = False
    for name, pool in settings.pools.items():
        scans.loc[scans.pool == name, "in_truth"] = scans.volume[scans.pool == name].isin(truth_volumes(settings.ground_truth, pool))
    return scans, held.count_rows("pool != 'gt'")


def funnel(scans: pd.DataFrame, total: int) -> pd.DataFrame:
    """The scans taken, printing how many each condition leaves."""
    def step(name: str, rows: pd.Series) -> pd.Series:
        print(f"{name:52} {int(rows.sum()):10,}  {rows.sum() / len(scans):6.1%}"); return rows
    print(f"{'pool scans':52} {total:10,}\n{'hold the columns (embedded)':52} {len(scans):10,}  of them:")
    labelled = step("lines labelled by the old pipeline", (scans.lines > 0) & (scans.source == "htrflow"))
    cut = step("... the head says an opening to cut", labelled & (scans.cut_class == "cut"))
    text = step("... with running text, no table", cut & ~scans.is_table)
    text = step("... and handwriting, not print or type", text & scans.kind.isin(HANDWRITING))
    placed = step("... a fold is placed", text & scans.fold_found.notna())
    found = step("... a fold line is found", placed & (scans.fold_found == True))  # noqa: E712 (a column with nulls)
    taken = step("... outside the ground truth's volumes: TAKEN, cut", found & ~scans.in_truth)
    single = step("single pages with running text, outside the truth", labelled & (scans.cut_class == "single") & ~scans.is_table & scans.kind.isin(HANDWRITING) & ~scans.in_truth)
    pages = round(taken.sum() * 1.847) + (int(single.sum()) if SINGLES else 0)  # the first trusted set gave 1.847 pages a cut scan
    print(f"{'pages, at 1.847 a cut scan as in the first set':52} {pages:10,}" + (" (with the single pages)" if SINGLES else ""))
    step("of the taken: colour scans", taken & scans.colour.fillna(False).astype(bool))
    tables = labelled & scans.is_table
    step("the other filter: tables", tables)
    for name in ("cut", "no_cut", "single", "odd", "not_a_page"):
        step(f"    tables the head calls {name}", tables & (scans.cut_class == name))
    return scans[taken | (single if SINGLES else False)]


def stage(out: Path, row, settings: CurateSettings) -> bool:
    """Links to one scan and its label file below ``out``; False when either is missing."""
    labels = settings.pools[row.pool].label_file(row.scan); image = Path(row.image)
    if not labels.exists() or not image.exists():
        return False
    folder = out / row.volume; (folder / "alto").mkdir(parents=True, exist_ok=True)
    for link, target in ((folder / "alto" / labels.name, labels), (folder / image.name, image)):
        if not link.is_symlink():
            link.symlink_to(target)
    return True


if __name__ == "__main__":
    scans, total = pools(); taken = funnel(scans, total)
    if sys.argv[1] != "count":
        out = Path(sys.argv[1]); settings = CurateSettings()
        if os.environ.get("EXCLUDE"):  # scans other staged sets hold already (their scans.parquet, comma-separated)
            held = set().union(*(pd.read_parquet(path, columns=["scan"]).scan for path in os.environ["EXCLUDE"].split(",")))
            taken = taken[~taken.scan.isin(held)]; print(len(held), "scans in other sets left out;", len(taken), "left")
        if len(sys.argv) > 2:
            taken = taken.sample(min(int(sys.argv[2]), len(taken)), random_state=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
        assert taken.scan.is_unique  # cuts.json names a scan by its file name
        with ThreadPoolExecutor(32) as threads:
            staged = list(threads.map(lambda row: stage(out, row, settings), taken.itertuples()))
        taken = taken[staged]
        cuts = {row.scan: float(row.fold_x) for row in taken.itertuples() if row.cut_class == "cut"}
        (out / "cuts.json").write_text(json.dumps(cuts), encoding="utf-8"); taken.to_parquet(out / "scans.parquet")
        print(len(taken), "scans staged in", out, "|", len(cuts), "to cut |", taken.pool.value_counts().to_dict())
