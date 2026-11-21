"""The pages the orientation network calls turned (over 0.9), turned back and staged as a set to read again (CURATION E66).

    uv run python runs/htr/curate/prototypes/turned_reread.py stage    # images turned back, checked, set written

From trusted_v1 and pool_v2. A page is turned by the opposite of the turn the network saw (`turn` is the counter-
clockwise turn of the page from upright, np.rot90's sense), saved as <id>.jpg in /mnt/scratch/gabriel/htr/turned_v1,
and seen by the network again: a page it does not now call upright is left out. The set's rows keep the page's id,
its meta and its old label (unused: the reader's reading becomes the label).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
sys.path.insert(0, "runs/htr/curate/prototypes")
from orientation_pages import SIDE, block
import torch.nn as nn, torch.nn.functional as F
Image.MAX_IMAGE_PIXELS = None
HOME = Path("runs/htr/curate"); S = Path("/mnt/scratch/gabriel/htr"); OUT = S / "turned_v1"; (OUT / "pages_pages").mkdir(parents=True, exist_ok=True)
net = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 96), block(96, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 4))
net.load_state_dict(torch.load(HOME / "models/orientation.pt", weights_only=True)); net.eval()
def seen(image: Image.Image) -> tuple[int, float]:
    x = torch.from_numpy(np.asarray(image.convert("L").resize((SIDE, SIDE), Image.BILINEAR), dtype=np.uint8)).float().div(255)[None, None]
    with torch.inference_mode(): p = F.softmax(net(x), 1)[0]
    return int(p.argmax()) * 90, float(p.max())
rows, metas, kept, lost = [], [], 0, 0
for name in ("trusted_v1", "pool_v2"):
    o = pd.read_parquet(HOME / f"orientation/{name}.parquet"); turned = o[(o.turn != 0) & (o.p > 0.9)].set_index("id")
    stem = S / f"{name}_nemotron/pages"
    with open(f"{stem}.jsonl", encoding="utf-8") as f, open(f"{stem}.meta.jsonl", encoding="utf-8") as m:
        for line, meta_line in zip(f, m):
            meta = json.loads(meta_line)
            if meta["id"] not in turned.index: continue
            row = json.loads(line); turn = int(turned.turn[meta["id"]])
            with Image.open(row["images"][0]) as image: back = image.convert("RGB").rotate(-turn, expand=True)  # undo a counter-clockwise turn
            again, p = seen(back)
            if again != 0: lost += 1; continue
            path = OUT / "pages_pages" / f"{meta['id']}.jpg"; back.save(path, quality=92)
            rows.append(json.dumps({**row, "images": [str(path)]}, ensure_ascii=False)); metas.append(json.dumps({**meta, "set": name, "turned_back": turn}, ensure_ascii=False)); kept += 1
(OUT / "pages.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8"); (OUT / "pages.meta.jsonl").write_text("\n".join(metas) + "\n", encoding="utf-8")
print(kept, "pages turned back and seen upright;", lost, "the network does not call upright when turned back (left out)")
