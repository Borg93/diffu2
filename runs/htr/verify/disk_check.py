"""After the file server's failure of 19 Nov 2026: is what we keep on /mnt/scratch and /mnt/work whole?

    uv run --with rapidfuzz python runs/htr/verify/disk_check.py

Each check prints OK or PROBLEM with numbers. Reads only.
"""
import glob, hashlib, json, os, random, sys, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import lance, numpy as np, pyarrow.parquet as pq
from PIL import Image
Image.MAX_IMAGE_PIXELS = None; S = "/mnt/scratch/gabriel"; W = "/mnt/work/gabriel"; rng = random.Random(0)
def say(ok, what): print(("OK      " if ok else "PROBLEM ") + what, flush=True)
def decodes(path):
    try:
        with Image.open(path) as im: im.draft("L", (400, 400)); im.load(); return True
    except Exception: return False
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""): h.update(block)
    return h.hexdigest()

# 1. the draft and the base repository: recorded hashes
m = json.load(open(f"{S}/models/nemotron-3.5-parse-repo/dflash2/manifest.json"))
for name, want in m["files"].items(): say(sha(f"{S}/models/nemotron-3.5-parse-repo/dflash2/{name}") == want["sha256"], f"dflash2/{name}: sha256 as its manifest records")

# 2. every safetensors and .pt model file on the disks: header reads, size matches the header
from safetensors import safe_open
bad = []; files = [p for root in (f"{S}/models", f"{W}/models") for p in glob.glob(root + "/**/*.safetensors", recursive=True)]
for p in files:
    try:
        with open(p, "rb") as f: n = int.from_bytes(f.read(8), "little"); head = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in head.items() if k != "__metadata__")
        if os.path.getsize(p) != 8 + n + end: bad.append(p)
    except Exception as e: bad.append(f"{p}: {type(e).__name__}")
say(not bad, f"{len(files)} safetensors files under models/: header parses and the file is as long as its header says" + (f"; bad: {bad[:5]}" if bad else ""))

# 3. the page sets: lines, JSON, images present, a sample decodes
def page_set(stem, want, sample=4000):
    rows = open(stem + ".jsonl", encoding="utf-8").read().splitlines(); metas = open(stem + ".meta.jsonl", encoding="utf-8").read().splitlines()
    try: images = [json.loads(r)["images"][0] for r in rows]; [json.loads(x) for x in metas[:: max(1, len(metas) // 20000)]]; parsed = True
    except Exception: images = []; parsed = False
    with ThreadPoolExecutor(32) as pool:
        present = sum(pool.map(os.path.exists, images)); pick = rng.sample(images, min(sample, len(images))); good = sum(pool.map(decodes, pick))
    say(parsed and len(rows) == want == len(metas) and present == len(images) and good == len(pick), f"{stem}: {len(rows)} rows (expected {want}), meta {len(metas)}, images present {present}, {good} of {len(pick)} sampled images decode")
page_set(f"{S}/htr/pool_v2_nemotron/pages", 999808); page_set(f"{S}/htr/trusted_v1_nemotron/pages", 139937)
page_set(f"{W}/htr/nemotron_parse_pages", 49491); page_set(f"{W}/htr/nemotron_parse_pages.dev", 974); page_set(f"{W}/htr/nemotron_parse_pages.test", 2971)

# 4. the vectors: every part reads in full; rows, NaN, norms; and they still give the pca column computed before the failure
kept = np.load("runs/htr/curate/heads/pca.npz"); held = lance.dataset("runs/htr/curate/scans.lance"); total = 0; trouble = []; worst = 0.0; seen = set()
for part in sorted(glob.glob(f"{S}/htr/embed_pool/part-*.parquet")):
    try:
        t = pq.read_table(part); n = len(t); total += n; images = t.column("image").to_pylist(); seen.update(images)
        X = np.empty((n, 4096), np.float32)
        for at, name in ((0, "tipsv2"), (2048, "dinov3")):
            X[:, at:at + 2048] = np.concatenate([c.flatten().to_numpy(zero_copy_only=False) for c in t.column(name).chunks]).reshape(n, 2048)
        if not np.isfinite(X).all() or abs(float(np.linalg.norm(X[:, :2048], axis=1).mean()) - 1.414) > 0.01: trouble.append(os.path.basename(part) + " values")
        pickrows = rng.sample(range(n), 40); X = X[pickrows]
        for at in (0, 2048): X[:, at:at + 2048] /= np.linalg.norm(X[:, at:at + 2048], axis=1, keepdims=True)
        mine = ((X - kept["mean"]) @ kept["components"].T).astype(np.float16).astype(np.float32)
        names = [images[k] for k in pickrows]; got = held.to_table(columns=["image", "pca"], filter="image IN (" + ",".join("'" + x.replace("'", "''") + "'" for x in names) + ")").to_pandas().set_index("image")
        worst = max(worst, max(float(np.abs(mine[k] - np.asarray(got.pca[names[k]], np.float32)).max()) for k in range(len(names))))
    except Exception as e: trouble.append(f"{os.path.basename(part)}: {type(e).__name__} {e}"[:120])
say(not trouble and total == 3068911 == len(seen) and worst < 0.02, f"embed_pool: {total} rows in {len(glob.glob(f'{S}/htr/embed_pool/part-*.parquet'))} parts, {len(seen)} different scans (expected 3,068,911); 40 scans a part recomputed against the pca column stored before the failure: largest difference {worst:.4f}" + (f"; trouble: {trouble[:5]}" if trouble else ""))

# 5. the pools: scans and label files exist and read; a label's line count is the one the table holds
from diffu_curate.settings import CurateSettings
settings = CurateSettings(); t = held.to_table(columns=["image", "pool", "scan", "lines"], filter="pool != 'gt' AND lines > 0").to_pandas().sample(6000, random_state=0)
def one(row):
    labels = settings.pools[row.pool].label_file(row.scan)
    if not os.path.exists(row.image) or not os.path.exists(labels): return "missing"
    if not decodes(row.image): return "image"
    try: n = sum(1 for _ in ET.parse(labels).getroot().iter("{http://www.loc.gov/standards/alto/ns-v4#}TextLine"))
    except Exception: return "label"
    return "same" if n == row.lines else "lines"
with ThreadPoolExecutor(24) as pool: got = list(pool.map(one, t.itertuples()))
counts = {k: got.count(k) for k in sorted(set(got))}; say(counts.get("same", 0) >= 0.995 * len(got) and not counts.get("missing") and not counts.get("image") and not counts.get("label"), f"pools: 6,000 scans at random with their label files: {counts} ('same': the scan decodes and its label holds the number of lines the table recorded)")
for name, pool in settings.pools.items():
    volumes = held.to_table(columns=["volume"], filter=f"pool = '{name}'").column("volume").unique().to_pylist(); there = sum(os.path.isdir(os.path.dirname(settings.pools[name].label_file(v + "_00001"))) for v in volumes)
    say(there == len(volumes), f"pool {name}: {there} of {len(volumes)} label folders present")
print("done", flush=True)
