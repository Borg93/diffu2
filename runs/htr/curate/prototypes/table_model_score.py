"""NVIDIA's table-structure model scored on the ground truth's own tables (CURATION E45), on the CPU.

The ground truth marks a table with rows and cells on 139 scans of the training and dev sets (232 table regions, a
median of 20 rows and 3 columns). Each region is cut out of its scan and given to Nemotron Table Structure v1; the
rows and columns it finds (scored above SURE) are matched one to one with the truth's by how far their spans overlap
(the rows' in y, the columns' in x; a match from 0.5). A truth row is the span of the cells of that row that span
one row, a truth column likewise.

    CUDA_VISIBLE_DEVICES= uv run --with matplotlib python runs/htr/curate/prototypes/table_model_score.py OUT
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from diffu_curate.settings import CurateSettings
from table_models_look import MODELS, TableExp, TableWrapper, load, postprocess_preds_table_structure

LEVELS = (0.05, 0.1, 0.2, 0.3, 0.5)  # the model's scores a box is taken from; 0.3 is the level of the tables printed

Image.MAX_IMAGE_PIXELS = None
CELL = re.compile(r'<TableCell row="(\d+)" col="(\d+)" rowSpan="(\d+)" colSpan="(\d+)"[^>]*>\s*<Coords points="([^"]+)"')


def box(points: str) -> tuple[float, float, float, float]:
    xy = np.array([[float(v) for v in point.split(",")] for point in points.split()])
    return xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()


def spans(cells: list[tuple[int, int, int, int, tuple]], axis: int) -> list[tuple[float, float]]:
    """The truth's rows (axis 0: spans in y) or columns (axis 1: spans in x), from the cells that span one of them."""
    out = {}
    for row, col, row_span, col_span, (x0, y0, x1, y1) in cells:
        if (row_span, col_span)[axis] != 1:
            continue
        low, high = (y0, y1) if axis == 0 else (x0, x1)
        held = out.get((row, col)[axis]); out[(row, col)[axis]] = (low, high) if held is None else (min(held[0], low), max(held[1], high))
    return [out[key] for key in sorted(out)]


def matched(truth: list[tuple[float, float]], found: list[tuple[float, float]]) -> int:
    """How many truth spans pair off one to one with a found span they overlap by 0.5 or more (intersection over union)."""
    pairs = sorted(((min(a[1], b[1]) - max(a[0], b[0])) / (max(a[1], b[1]) - min(a[0], b[0]) + 1e-6), i, j)
                   for i, a in enumerate(truth) for j, b in enumerate(found))
    used_truth, used_found = set(), set()
    for overlap, i, j in reversed(pairs):
        if overlap < 0.5:
            break
        if i not in used_truth and j not in used_found:
            used_truth.add(i); used_found.add(j)
    return len(used_truth)


if __name__ == "__main__":
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES= (cuda:0 is another user's GPU)"
    torch.set_num_threads(8)
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    model = load(TableExp, TableWrapper, MODELS / "nemotron-table-structure-v1/nemotron_table_structure_v1/weights.pth")
    truth_root = CurateSettings().ground_truth; scans = {}
    for split in ("train", "dev"):
        targets = open(f"data/htr/nemotron_parse_pages.{split}.jsonl", encoding="utf-8")
        for target, meta in zip(targets, open(f"data/htr/nemotron_parse_pages.{split}.meta.jsonl", encoding="utf-8")):
            if "begin{tabular}" in target:
                page = json.loads(meta); scans[(page["collection"], page["volume"], page["scan"])] = split
    results = []; by_level: dict[float, np.ndarray] = {}
    for (collection, volume, scan), split in sorted(scans.items()):
        labels = next(truth_root.glob(f"{collection}/*/*/page/{Path(scan).stem}.xml"))
        image = np.asarray(Image.open(labels.parent.parent / scan).convert("RGB"))
        for n, region in enumerate(re.findall(r"<TableRegion.*?</TableRegion>", labels.read_text(encoding="utf-8"), re.S)):
            cells = [(int(r), int(c), int(rs), int(cs), box(points)) for r, c, rs, cs, points in CELL.findall(region)]
            rows, columns = spans(cells, 0), spans(cells, 1)
            if len(rows) < 2 or len(columns) < 2:
                continue
            x0 = int(max(0, min(cell[4][0] for cell in cells))); y0 = int(max(0, min(cell[4][1] for cell in cells)))
            x1 = int(min(image.shape[1], max(cell[4][2] for cell in cells))); y1 = int(min(image.shape[0], max(cell[4][3] for cell in cells)))
            if x1 - x0 < 50 or y1 - y0 < 50:
                continue
            cut = np.ascontiguousarray(image[y0:y1, x0:x1])
            with torch.inference_mode():
                raw = model(model.preprocess(cut), cut.shape)[0]
            boxes, kinds, scores = postprocess_preds_table_structure(raw, min(LEVELS), model.labels)
            boxes = boxes * np.array([cut.shape[1], cut.shape[0], cut.shape[1], cut.shape[0]]) if len(boxes) else np.zeros((0, 4))
            for level in LEVELS:
                keep_rows = boxes[(kinds == model.labels.index("row")) & (scores > level)]; keep_columns = boxes[(kinds == model.labels.index("column")) & (scores > level)]
                tally = by_level.setdefault(level, np.zeros(6))
                tally += [len(rows), len(keep_rows), matched(rows, [(b[1] + y0, b[3] + y0) for b in keep_rows]),
                          len(columns), len(keep_columns), matched(columns, [(b[0] + x0, b[2] + x0) for b in keep_columns])]
            found = {name: boxes[(kinds == model.labels.index(name)) & (scores > 0.3)] for name in ("row", "column", "cell")}
            found_rows = [(b[1] + y0, b[3] + y0) for b in found["row"]]; found_columns = [(b[0] + x0, b[2] + x0) for b in found["column"]]
            results.append({"collection": collection, "scan": scan, "table": n, "split": split, "rows": len(rows), "rows found": len(found_rows),
                            "rows matched": matched(rows, found_rows), "columns": len(columns), "columns found": len(found_columns),
                            "columns matched": matched(columns, found_columns), "cells": len(cells), "cells found": len(found["cell"])})
            if len(results) <= 12:  # the first tables drawn: the truth's rows and columns green, the model's blue and orange
                shown = Image.fromarray(cut); pen = ImageDraw.Draw(shown); w = max(2, cut.shape[1] // 300)
                for low, high in rows: pen.line((0, low - y0, cut.shape[1], low - y0), fill=(20, 160, 60), width=w)
                for low, high in columns: pen.line((low - x0, 0, low - x0, cut.shape[0]), fill=(20, 160, 60), width=w)
                for b in found["row"]: pen.rectangle(tuple(b), outline=(40, 110, 220), width=w)
                for b in found["column"]: pen.rectangle(tuple(b), outline=(240, 140, 20), width=w)
                shown.thumbnail((700, 700)); shown.save(out / f"table{len(results):02}.jpg", quality=85)
            print(results[-1], flush=True)
    table = pd.DataFrame(results); table.to_csv(out / "scores.csv", index=False)
    def rates(part: pd.DataFrame) -> pd.Series:
        return pd.Series({"tables": len(part), "rows found of the truth's": part["rows matched"].sum() / part.rows.sum(),
                          "found rows that are right": part["rows matched"].sum() / max(part["rows found"].sum(), 1),
                          "columns found of the truth's": part["columns matched"].sum() / part["columns"].sum(),
                          "found columns that are right": part["columns matched"].sum() / max(part["columns found"].sum(), 1)})
    print("\n== all tables"); print(rates(table).round(3).to_string())
    print("\n== by collection"); print(table.groupby("collection").apply(rates).round(3).sort_values("tables", ascending=False).to_string())
    print("\n== by the score a box is taken from: rows found, found rows right, columns found, found columns right")
    for level, (n_rows, got_rows, hit_rows, n_columns, got_columns, hit_columns) in by_level.items():
        print(f"{level:5}  {hit_rows / n_rows:.3f}  {hit_rows / max(got_rows, 1):.3f}  {hit_columns / n_columns:.3f}  {hit_columns / max(got_columns, 1):.3f}")
    print("\ntables with every column found:", round(float((table["columns matched"] == table["columns"]).mean()), 3),
          "| with 90 % of their rows found:", round(float((table["rows matched"] >= 0.9 * table.rows).mean()), 3))
