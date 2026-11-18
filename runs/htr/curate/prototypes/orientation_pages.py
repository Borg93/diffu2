"""Which way up the writing is on every page of a reader set, by E6's small network, on the CPU (CURATION E58).

    uv run python runs/htr/curate/prototypes/orientation_pages.py SET_STEM OUT.parquet [LIMIT]

A page image (the single page the reader is given) as a 256-pixel square greyscale thumbnail through the network of
orientation.py (runs/htr/curate/models/orientation.pt). Rows: id, image, turn (0, 90, 180 or 270 degrees, counter-
clockwise, the turn the network sees), p (its probability), written in parts of 20,000 beside OUT; resumable.
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
Image.MAX_IMAGE_PIXELS = None; SIDE = 256; PART = 20000; torch.set_num_threads(int(os.environ.get("THREADS", "8")))

def thumb(path):
    try:
        with Image.open(path) as im:
            im.draft("L", (SIDE * 4, SIDE * 4)); return np.asarray(im.convert("L").resize((SIDE, SIDE), Image.BILINEAR), dtype=np.uint8)
    except (OSError, ValueError):
        return None

def block(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))

if __name__ == "__main__":
    stem, out = sys.argv[1], Path(sys.argv[2]); limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    parts = out.with_suffix(""); parts.mkdir(parents=True, exist_ok=True)
    net = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 96), block(96, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 4))
    net.load_state_dict(torch.load("runs/htr/curate/models/orientation.pt", weights_only=True)); net.eval()
    metas = [json.loads(l) for l in open(f"{stem}.meta.jsonl", encoding="utf-8")]
    done = {i for part in parts.glob("part-*.parquet") for i in pq.read_table(part, columns=["id"]).column("id").to_pylist()}
    todo = []
    with open(f"{stem}.jsonl", encoding="utf-8") as rows:
        for meta, row in zip(metas, rows):
            if meta["id"] not in done: todo.append((meta["id"], json.loads(row)["images"][0]))
    todo = todo[:limit]; print(len(metas), "pages;", len(done), "done before;", len(todo), "to do", flush=True); start = time.time()
    with Pool(int(os.environ.get("WORKERS", "24"))) as pool:
        for at in range(0, len(todo), PART):
            chunk = todo[at:at + PART]; images = pool.map(thumb, [path for _, path in chunk], chunksize=64)
            keep = [k for k, image in enumerate(images) if image is not None]
            with torch.inference_mode():
                p = torch.cat([F.softmax(net(torch.from_numpy(np.stack([images[k] for k in keep[b:b + 256]])).float().div(255).unsqueeze(1)), 1) for b in range(0, len(keep), 256)])
            table = pa.table({"id": [chunk[k][0] for k in keep], "image": [chunk[k][1] for k in keep],
                              "turn": pa.array((p.argmax(1) * 90).tolist(), pa.int16()), "p": pa.array(p.max(1).values.tolist(), pa.float32())})
            part = parts / f"part-{time.time_ns()}.parquet"; pq.write_table(table, part.with_suffix(".unfinished")); part.with_suffix(".unfinished").replace(part)
            print(f"{at + len(chunk)} of {len(todo)}, {(at + len(chunk)) / (time.time() - start):.0f} pages a second, {len(chunk) - len(keep)} unreadable", flush=True)
    pq.write_table(pa.concat_tables([pq.read_table(part) for part in sorted(parts.glob("part-*.parquet"))]), out); print("done", flush=True)
