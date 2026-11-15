"""The GT pages for tuning the writing layer: split by volume into train / val / test, each page with its rectangle on
the scan, its lines' PAGE polygons in page coordinates, and what kind of page it is (a table region on it, digit share,
colour or greyscale scan)."""
import json, random, collections
from pathlib import Path
import xml.etree.ElementTree as ET
import lance, numpy as np
from PIL import Image
from diffu_page.page.layout.xml_replay import load_xml_layout
Image.MAX_IMAGE_PIXELS = None
OUT = Path("runs/htr/curate/writing_tune")
t = lance.dataset("runs/htr/curate/pages.lance").to_table(columns=["page", "collection", "volume", "labels", "image", "x0", "y0", "x1", "y1", "lines", "split"], filter="pool = 'gt' AND lines > 0").to_pylist()
t = [r for r in t if r["split"] == "train"]  # the GT's own test pages stay untouched
vols = sorted({(r["collection"], r["volume"]) for r in t}); random.Random(0).shuffle(vols)
role = {v: ("test" if i % 10 == 0 else "val" if i % 10 == 1 else "train") for i, v in enumerate(vols)}
by = collections.defaultdict(list)
for r in t: by[role[(r["collection"], r["volume"])]].append(r)
want = {"train": (1200, 400), "val": (300, 100), "test": (300, 100)}  # (prose, table/ledger) pages a role
def kind(r):
    root = ET.parse(r["labels"]).getroot()
    table = any(e.tag.endswith("TableRegion") for e in root.iter())
    layout = load_xml_layout(Path(r["labels"]))
    lines = [l for l in layout.lines if r["x0"] <= l.middle_x < r["x1"]]
    text = "".join(l.text for l in lines); digits = sum(c.isdigit() for c in text) / max(1, len(text))
    with Image.open(r["image"]) as im:
        im.draft("RGB", (256, 256)); a = np.asarray(im.convert("RGB").resize((64, 64))).astype(int)
    colour = bool(np.abs(a[..., 0] - a[..., 2]).mean() > 4)
    polys = [[(x - r["x0"], y - r["y0"]) for x, y in l.polygon] if len(l.polygon) >= 3 else [(l.box[0] - r["x0"], l.box[1] - r["y0"]), (l.box[2] - r["x0"], l.box[1] - r["y0"]), (l.box[2] - r["x0"], l.box[3] - r["y0"]), (l.box[0] - r["x0"], l.box[3] - r["y0"])] for l in lines]
    return {"table": table, "digits": round(digits, 3), "colour": colour, "polygons": polys, "n_poly": sum(len(l.polygon) >= 3 for l in lines), "n_lines": len(lines)}
out = []
from multiprocessing import Pool
def judged(r):
    try:
        return {**{x: r[x] for x in ("page", "collection", "volume", "image", "x0", "y0", "x1", "y1")}, **kind(r)}
    except Exception as e:  # a page whose files cannot be read is left out, and counted
        return None
for split, (n_prose, n_table) in want.items():
    rows = by[split]; random.Random(1).shuffle(rows)
    with Pool(24) as p: judged_rows = [x for x in p.map(judged, rows[:12000], chunksize=32) if x is not None]
    ledger = [x for x in judged_rows if x["table"] or x["digits"] > 0.15]; prose = [x for x in judged_rows if not (x["table"] or x["digits"] > 0.15)]
    print(split, "judged", len(judged_rows), "ledger/table", len(ledger), "prose", len(prose), flush=True)
    for x in prose[:n_prose] + ledger[:n_table]: out.append({**x, "role": split})
json.dump(out, open(OUT / "pages.json", "w"))
import pandas as pd
d = pd.DataFrame(out); d["kind"] = np.where(d.table | (d.digits > 0.15), "table/ledger", "prose") + np.where(d.colour, " colour", " greyscale")
pd.set_option("display.width", 200); print(d.groupby(["role", "kind"]).size().unstack(fill_value=0).to_string()); print("lines with real polygons: %.3f" % (d.n_poly.sum() / d.n_lines.sum()), "| volumes:", d.groupby("role").volume.nunique().to_dict())
