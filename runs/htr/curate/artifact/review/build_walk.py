"""One real scan through the pipeline, for the review page's walkthrough (walk.json beside index.html)."""
import os
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import base64, io, json, re, sys
from pathlib import Path
import lance, pandas as pd
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
HOME = Path("runs/htr/curate"); S = Path("/mnt/scratch/gabriel/htr"); out = Path(sys.argv[1])
BLOCK = re.compile(r"<x_([0-9.]+)><y_([0-9.]+)>(.*?)<x_([0-9.]+)><y_([0-9.]+)><class_([A-Za-z-]+)>", re.DOTALL)
def uri(im, side):
    im = im.copy(); im.thumbnail((side, side)); buf = io.BytesIO(); im.save(buf, "JPEG", quality=78); return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
d = pd.read_parquet(HOME / "disagreement/pool_v2.parquet")
metas = {json.loads(l)["id"]: json.loads(l) for l in open(S / "pool_v2_nemotron/pages.meta.jsonl", encoding="utf-8")}
d["scan"] = d.id.map(lambda i: metas[i]["scan"].rsplit(".", 1)[0]); d["side"] = d.id.map(lambda i: metas[i]["side"])
both = d[(d.finish == "stop") & d.disagreement.between(0.03, 0.08) & (d.label_chars.between(700, 1100))].groupby("scan").filter(lambda g: len(g) == 2)
scan = both.scan.drop_duplicates().sample(1, random_state=4).iloc[0]; pages = both[both.scan == scan].set_index("side")
row = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", "cut_class", "cut_p", "is_table", "table_p", "kind", "language", "colour", "fold_x", "pool_cluster_name"], filter=f"scan = '{scan}'").to_pylist()[0]
with Image.open(row["image"]) as im:
    im.draft("RGB", (1800, 1800)); im = im.convert("RGB")
pen = ImageDraw.Draw(im); x = row["fold_x"] * im.width; pen.line([(x, 0), (x, im.height)], fill=(214, 48, 39), width=max(4, im.width // 250))
want = set(pages.id); raw, rows = {}, {}
with open(HOME / "pool_read/pool_v2_vllm.jsonl", encoding="utf-8") as f:
    for line in f:
        i = line[8:60].split('"')[0]
        if i in want: raw[i] = json.loads(line)["raw"]
with open(S / "pool_v2_nemotron/pages.jsonl", encoding="utf-8") as f, open(S / "pool_v2_nemotron/pages.meta.jsonl", encoding="utf-8") as m:
    for line, ml in zip(f, m):
        if json.loads(ml)["id"] in want: rows[json.loads(ml)["id"]] = json.loads(line)
def lines(text, n=5):
    b = BLOCK.findall(text); return [l[:80] for l in max(b, key=lambda x: len(x[2]))[2].split("\n") if l.strip()][:n] if b else []
walk = {"scan": scan, "image": uri(im, 900), "columns": {k: row[k] for k in ("cut_class", "cut_p", "is_table", "table_p", "kind", "language", "colour", "fold_x", "pool_cluster_name")}, "pages": {}}
for side, r in pages.iterrows():
    page = rows[r.id]
    with Image.open(page["images"][0]) as pim: walk["pages"][side] = {"image": uri(pim.convert("RGB"), 520), "label": lines(page["messages"][1]["content"]), "reader": lines(raw[r.id]), "disagreement": float(r.disagreement), "label_chars": int(r.label_chars)}
json.dump(walk, open(out, "w"), ensure_ascii=False); print(scan, walk["columns"], {s: round(p["disagreement"], 3) for s, p in walk["pages"].items()}, round(out.stat().st_size / 1e3), "kB")
