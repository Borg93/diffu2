"""The trusted pages with our reader's reading as their label in place of the old pipeline's (CURATION E60).

    uv run python runs/htr/curate/prototypes/selftrain_set.py OUT_STEM

The pages of trusted_v1 (E38's pretraining set, 139,937 pages), each answered with nemotron-parse-more's reading
of it through vLLM (pool_read/trusted_v1_vllm.jsonl) instead of the old label. Left out: a reading stopped as a loop
or empty. A page called turned is given its image turned back and its reading of that (pool_read/turned_v1_vllm.jsonl,
E66), or is left out if it has none. Everything else as E38's set, so the two pretrainings differ in their labels.
"""
import json, sys
from pathlib import Path
import pandas as pd

HOME = Path("runs/htr/curate"); S = Path("/mnt/scratch/gabriel/htr"); out = Path(sys.argv[1])
def readings(path: Path) -> dict[str, str]:
    got = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("finish") == "stop" and r["raw"].strip(): got[r["id"]] = r["raw"].strip()
    return got
read = readings(HOME / "pool_read/trusted_v1_vllm.jsonl")
back = readings(HOME / "pool_read/turned_v1_vllm.jsonl") if (HOME / "pool_read/turned_v1_vllm.jsonl").exists() else {}
turned_image = {json.loads(m)["id"]: json.loads(r)["images"][0] for r, m in zip(open(S / "turned_v1/pages.jsonl"), open(S / "turned_v1/pages.meta.jsonl"))}
o = pd.read_parquet(HOME / "orientation/trusted_v1.parquet"); turned = set(o[(o.turn != 0) & (o.p > 0.9)].id)
rows, metas, counts = [], [], {"kept": 0, "turned back": 0, "loop or empty": 0, "turned, no reading": 0}
stem = S / "trusted_v1_nemotron/pages"
for line, meta_line in zip(open(f"{stem}.jsonl", encoding="utf-8"), open(f"{stem}.meta.jsonl", encoding="utf-8")):
    row, meta = json.loads(line), json.loads(meta_line); page = meta["id"]
    if page in turned:
        if page not in back or page not in turned_image: counts["turned, no reading"] += 1; continue
        row["images"] = [turned_image[page]]; row["messages"][1]["content"] = back[page]; counts["turned back"] += 1
    elif page in read:
        row["messages"][1]["content"] = read[page]
    else:
        counts["loop or empty"] += 1; continue
    counts["kept"] += 1; rows.append(json.dumps(row, ensure_ascii=False)); metas.append(meta_line.rstrip("\n"))
out.parent.mkdir(parents=True, exist_ok=True)
Path(f"{out}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8"); Path(f"{out}.meta.jsonl").write_text("\n".join(metas) + "\n", encoding="utf-8")
print(counts, "->", f"{out}.jsonl")
