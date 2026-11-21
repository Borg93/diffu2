"""Samples of the pages and scans we are unsure of, for the review page (samples.json beside index.html)."""
import os
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import base64, glob, io, json, re, sys
from pathlib import Path
import lance, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HOME = Path("runs/htr/curate"); S = Path("/mnt/scratch/gabriel/htr"); N = 8; out = Path(sys.argv[1])
BLOCK = re.compile(r"<x_([0-9.]+)><y_([0-9.]+)>(.*?)<x_([0-9.]+)><y_([0-9.]+)><class_([A-Za-z-]+)>", re.DOTALL)
def jpeg(path, side=340):
    with Image.open(path) as im:
        im.draft("RGB", (side * 2, side * 2)); im = im.convert("RGB"); im.thumbnail((side, int(side * 1.35)))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=70); return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
def lines(raw, n=4):
    blocks = BLOCK.findall(raw)
    if not blocks: return []
    return [l[:90] for l in max(blocks, key=lambda b: len(b[2]))[2].split("\n") if l.strip()][:n]

d = pd.read_parquet(HOME / "disagreement/pool_v2.parquet"); o = pd.read_parquet(HOME / "orientation/pool_v2.parquet").set_index("id")
metas = [json.loads(l) for l in open(S / "pool_v2_nemotron/pages.meta.jsonl", encoding="utf-8")]; at = {m["id"]: k for k, m in enumerate(metas)}
d["scan"] = d.id.map(lambda i: metas[at[i]]["scan"].rsplit(".", 1)[0]); d = d.join(o[["turn", "p"]], on="id")
s = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["scan", "image", "pool", "cut_class", "cut_p", "is_table", "table_p", "language", "kind", "pool_cluster_name"], filter="pool != 'gt' AND cut_class IS NOT NULL").to_pandas()
d = d.join(s.set_index("scan")[["cut_class", "language", "kind"]], on="scan")
good = (d.cut_class == "cut") & (d.finish == "stop") & ~((d.turn != 0) & (d.p > 0.9)) & (d.label_chars >= 100) & (d.language.isna() | (d.language == "swedish"))
PAGE_GROUPS = [
    ("pass", good & (d.disagreement <= 0.08) & (d.label_chars > 600)),
    ("near", good & d.disagreement.between(0.2, 0.3) & (d.label_chars > 400)),
    ("drop30", good & d.disagreement.between(0.3, 0.5) & (d.label_chars > 400)),
    ("drop50", good & (d.disagreement > 0.5) & (d.label_chars > 400)),
    ("loop", (d.cut_class == "cut") & (d.finish == "repetition")),
    ("turned", (d.turn != 0) & (d.p > 0.99)),
    ("foreign", (d.cut_class == "cut") & d.language.isin(["german", "latin"]) & (d.finish == "stop") & (d.label_chars > 400)),
    ("short", (d.cut_class == "cut") & (d.finish == "stop") & (d.label_chars < 100) & (d.label_chars > 0)),
]
picked = {key: d[mask].sample(min(N, int(mask.sum())), random_state=11) for key, mask in PAGE_GROUPS}
counts = {key: int(mask.sum()) for key, mask in PAGE_GROUPS}
want = set().union(*(set(p.id) for p in picked.values())); raw = {}
with open(HOME / "pool_read/pool_v2_vllm.jsonl", encoding="utf-8") as f:
    for line in f:
        i = line[8:60].split('"')[0]
        if i in want: raw[i] = json.loads(line)["raw"]
need = {at[i] for i in want}; rows = {}
with open(S / "pool_v2_nemotron/pages.jsonl", encoding="utf-8") as f:
    for k, line in enumerate(f):
        if k in need: rows[k] = json.loads(line)
groups = {}
for key, part in picked.items():
    items = []
    for r in part.itertuples():
        row = rows[at[r.id]]
        items.append({"img": jpeg(row["images"][0]), "label": lines(row["messages"][1]["content"]), "reader": lines(raw.get(r.id, "")),
                      "note": f"disagreement {r.disagreement:.0%}" + (f" · turned {int(r.turn)}°" if key == "turned" else "") + (f" · {r.language}" if key == "foreign" else "") + f" · {r.scan}"})
    groups[key] = {"count": counts[key], "of": len(d), "items": items}
# scan-level groups
old = lance.dataset(str(HOME / "scans.lance"), version=103).to_table(columns=["scan", "cut_class"], filter="pool != 'gt' AND cut_class = 'cut'").to_pandas()
w = pa.concat_tables([pq.read_table(p) for p in sorted(glob.glob(str(HOME / "writing/part-*.parquet")))]).to_pandas().merge(s[["image", "scan", "cut_class", "is_table", "kind"]], on="image")
wc = w[(w.cut_class == "cut") & ~w.is_table]
SCAN_GROUPS = [
    ("odd", s[(s.cut_class == "odd") & s.scan.isin(set(old.scan))], lambda r: f"P(cut) {r.cut_p:.2f}", len(s)),
    ("print", s[s.cut_class == "print"], lambda r: f"P(cut) {r.cut_p:.2f}", len(s)),
    ("table", s[s.is_table & (s.cut_class == "cut")], lambda r: f"P(table) {r.table_p:.2f}", len(s)),
    ("uncovered", wc[(wc.covered < 0.7) & (wc.written > 0.1)], lambda r: f"{r.covered:.0%} of the writing has a label", len(wc)),
    ("unfilled", wc[(wc.filled < 0.6) & (wc.labelled > 0.1)], lambda r: f"{r.filled:.0%} of the labelled area holds writing", len(wc)),
]
for key, part, note, of in SCAN_GROUPS:
    items = [{"img": jpeg(r.image, 420), "note": note(r) + f" · {r.scan}"} for r in part.sample(min(N, len(part)), random_state=11).itertuples()]
    groups[key] = {"count": len(part), "of": of, "items": items}
json.dump(groups, open(out, "w"), ensure_ascii=False); print({k: (v["count"], len(v["items"])) for k, v in groups.items()}, round(out.stat().st_size / 1e6, 1), "MB")
