"""The pool scans whose cut is trusted, staged for the set builder (CURATION E37, E38).

A scan is taken when the first form calls it an opening with no table across the fold, the defined question says
"cut" (vl_cut), the head on the marks says "cut" (head_cut), the placer finds a fold line (folds.lance), the old
pipeline labelled lines on it, and its volume holds no ground-truth scan. Below OUT each volume gets an ``alto/``
folder of links to the label files and, beside it, links to the scans, as ``diffu-reader-set --xml`` reads a ground
truth; ``OUT/cuts.json`` gives each scan's fold for ``--cuts``, and ``OUT/trusted.parquet`` the scans taken.

    uv run python runs/htr/curate/prototypes/stage_trusted.py OUT
"""
import json
import sys
from pathlib import Path

import lance

from diffu_curate.sample import truth_volumes
from diffu_curate.settings import CurateSettings

HOME = Path("runs/htr/curate")
RULE = ("pool != 'gt' AND vl_layout IN ('opening_both_pages_written', 'opening_one_page_empty') AND "
        "(vl_table_across_fold IS NULL OR vl_table_across_fold = false) AND vl_cut = 'cut' AND head_cut = 'cut' AND "
        "lines > 0 AND source = 'htrflow'")

if __name__ == "__main__":
    out = Path(sys.argv[1]); settings = CurateSettings()
    columns = ["image", "pool", "volume", "scan", "vl_kind", "vl_writing_layout", "head_p_cut"]
    scans = lance.dataset(str(HOME / "scans.lance")).to_table(columns=columns, filter=RULE).to_pandas()
    folds = lance.dataset(str(HOME / "folds.lance")).to_table(filter="fold_found = true").to_pandas().drop_duplicates("image", keep="last")
    scans = scans.merge(folds[["image", "fold_x"]], on="image")
    truth = {name: truth_volumes(settings.ground_truth, pool) for name, pool in settings.pools.items()}
    in_truth = [row.volume in truth[row.pool] for row in scans.itertuples()]
    print(len(scans), "scans pass the rule;", sum(in_truth), "left out, their volume holds ground truth")
    scans = scans[[not held for held in in_truth]]
    assert scans.scan.is_unique
    staged = []
    for row in scans.itertuples():
        pool = settings.pools[row.pool]; labels = pool.label_file(row.scan); image = Path(row.image)
        if not labels.exists() or not image.exists():
            continue
        folder = out / row.volume; (folder / "alto").mkdir(parents=True, exist_ok=True)
        for link, target in ((folder / "alto" / labels.name, labels), (folder / image.name, image)):
            if not link.is_symlink():
                link.symlink_to(target)
        staged.append(row.Index)
    scans = scans.loc[staged]
    (out / "cuts.json").write_text(json.dumps({row.scan: float(row.fold_x) for row in scans.itertuples()}), encoding="utf-8")
    scans.to_parquet(out / "trusted.parquet")
    print(len(scans), "scans staged;", scans.pool.value_counts().to_dict(), scans.vl_writing_layout.value_counts().to_dict(), scans.vl_kind.value_counts().to_dict())
