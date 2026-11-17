"""The hand marks, the placed folds and the map's clusters as columns of scans.lance (CURATION E30).

mark (cut, no_cut, single, not_a_page, odd: cut_head.marks, the fold clicks over the bulk marks), mark_fold_x (a
clicked fold), fold_x and fold_found (folds.lance), cluster, map_x, map_y (scan_map.lance). Run again after new marks
or folds: columns that exist are updated in place.

    PYTHONPATH=runs/htr/curate/prototypes uv run python runs/htr/curate/prototypes/marks_to_lance.py
"""
from pathlib import Path

import lance
import pyarrow as pa

from cut_head import marks
from diffu_curate.table import add_features

HOME = Path("runs/htr/curate")

if __name__ == "__main__":
    mark = marks()
    print(add_features(HOME / "scans.lance", pa.table({"image": mark.index.tolist(), "mark": mark.tolist()})), "marks")
    clicks = lance.dataset(str(HOME / "fold_marks.lance")).to_table().to_pandas().sort_values("marked_at").drop_duplicates("image", keep="last")
    clicks = clicks[clicks.mark == "fold"]
    print(add_features(HOME / "scans.lance", pa.table({"image": clicks.image.tolist(), "mark_fold_x": clicks.x.astype("float32").tolist()}, schema=pa.schema([("image", pa.utf8()), ("mark_fold_x", pa.float32())]))), "clicked folds")
    folds = lance.dataset(str(HOME / "folds.lance")).to_table(columns=["image", "fold_x", "fold_found"]).to_pandas().drop_duplicates("image", keep="last")
    print(add_features(HOME / "scans.lance", pa.table({"image": pa.array(folds.image, pa.utf8()), "fold_x": pa.array(folds.fold_x, pa.float32()), "fold_found": pa.array(folds.fold_found, pa.bool_())})), "folds")
    spot = lance.dataset(str(HOME / "scan_map.lance")).to_table().to_pandas().drop_duplicates("image", keep="last")
    print(add_features(HOME / "scans.lance", pa.table({"image": pa.array(spot.image, pa.utf8()), "map_x": pa.array(spot.map_x, pa.float32()), "map_y": pa.array(spot.map_y, pa.float32()), "cluster": pa.array(spot.cluster, pa.int32())})), "map places")
