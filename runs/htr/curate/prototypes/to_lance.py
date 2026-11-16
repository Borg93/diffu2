"""What the side runs wrote as files, joined to scans.lance as columns on the scan's image path (CURATION: one table).

The vision-language model's two passes (vl/*.jsonl) become typed ``vl_*`` columns plus ``vl``, its whole answer as
JSON; the full-resolution fold placer's rows (cut/placed_*.jsonl) become ``fold_*`` columns. Run again after a run
has added rows: the columns are replaced whole from every file found.

    uv run python runs/htr/curate/prototypes/to_lance.py
"""
import json
from pathlib import Path

import lance
import pyarrow as pa

from diffu_curate.table import add_features, json_column

HOME = Path("runs/htr/curate"); SCANS = HOME / "scans.lance"
FIRST = ["pilot.jsonl", "sample.jsonl", "gt_rest.jsonl", "sample2.jsonl"]
SECOND = ["sample_pass2.jsonl", "gt_rest_pass2.jsonl", "sample2_pass2.jsonl"]
KEPT = ("caption", "kind", "layout", "written_side", "text_columns", "table_across_fold", "writing_at_fold", "turned_writing", "fold_x")


def answers(names: list[str]) -> dict[str, dict]:
    """The last answer without an error for each image, over the files of ``names`` that exist."""
    said: dict[str, dict] = {}
    for name in names:
        if (HOME / "vl" / name).exists():
            for line in (HOME / "vl" / name).open(encoding="utf-8"):
                row = json.loads(line)
                if "error" not in row:
                    said[row["image"]] = row
    return said


def text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def flag(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def vl_features() -> pa.Table:
    first, second = answers(FIRST), answers(SECOND)
    images = sorted(first)
    whole = [{**{key: first[image].get(key) for key in KEPT}, **{key: second[image][key] for key in ("writing_layout", "turned") if image in second}} for image in images]
    columns = lambda row: row.get("text_columns") if isinstance(row.get("text_columns"), int) and not isinstance(row.get("text_columns"), bool) else None
    fold = lambda row: float(row["fold_x"]) if isinstance(row.get("fold_x"), (int, float)) and not isinstance(row.get("fold_x"), bool) else None
    return pa.table({
        "image": images,
        "vl_caption": pa.array([text(row.get("caption")) for row in whole], pa.utf8()),
        "vl_kind": pa.array([text(row.get("kind")) for row in whole], pa.utf8()),
        "vl_layout": pa.array([text(row.get("layout")) for row in whole], pa.utf8()),
        "vl_written_side": pa.array([text(row.get("written_side")) for row in whole], pa.utf8()),
        "vl_text_columns": pa.array([columns(row) for row in whole], pa.int32()),
        "vl_table_across_fold": pa.array([flag(row.get("table_across_fold")) for row in whole], pa.bool_()),
        "vl_writing_at_fold": pa.array([flag(row.get("writing_at_fold")) for row in whole], pa.bool_()),
        "vl_fold_x": pa.array([fold(row) for row in whole], pa.float32()),
        "vl_writing_layout": pa.array([text(row.get("writing_layout")) for row in whole], pa.utf8()),
        "vl_turned": pa.array([text(row.get("turned")) for row in whole], pa.utf8()),
        "vl": json_column(whole),
    })


def fold_features() -> pa.Table:
    placed: dict[str, dict] = {}
    for path in sorted((HOME / "cut").glob("placed_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            row = json.loads(line); placed[row["image"]] = row
    images = sorted(placed); number = lambda key, kind: pa.array([placed[image].get(key) for image in images], kind)
    return pa.table({"image": images, "fold_placed": number("placed", pa.float32()), "fold_net": number("net", pa.float32()),
                     "fold_placed_inside": number("placed_inside", pa.int32()), "fold_net_inside": number("net_inside", pa.int32()),
                     "fold_lines": number("lines", pa.int32()), "fold_gap": number("gap", pa.float32()), "fold_stands": number("stands", pa.float32())})


if __name__ == "__main__":
    for name, features in (("vl", vl_features()), ("fold", fold_features())):
        print(name, features.num_rows, "rows,", add_features(SCANS, features), "matched", flush=True)
    held = lance.dataset(str(SCANS))
    for column in ("vl_kind", "vl_layout", "vl_writing_layout", "vl_turned", "pool"):
        held.create_scalar_index(column, "BITMAP")
    held.create_scalar_index("image", "BTREE")
    print("version", held.version, "columns", len(held.schema.names))
