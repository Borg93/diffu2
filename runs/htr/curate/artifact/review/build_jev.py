"""Jev's verdict on single lines, for the review page: the line's image, the old label's text, Jev's score.
Run with envs/vllm's Python on GPU 2: writes jev.json beside index.html."""
import base64, glob, io, json, math, random, sys
from pathlib import Path
from PIL import Image
from vllm import LLM, PoolingParams
from diffu_htr_train.jev.prompt import model_prompt
from diffu_htr_train.jev.render import render_text
from diffu_htr_train.jev.score import FONT, alto_lines, line_crop
Image.MAX_IMAGE_PIXELS = None
def item(score, text, crop, scan):
    crop = crop.copy(); crop.thumbnail((900, 48)); buf = io.BytesIO(); crop.save(buf, "JPEG", quality=85)
    return {"img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "text": text, "score": round(score, 3), "scan": scan}


if __name__ == "__main__":
    scans = [json.loads(l) for p in glob.glob("runs/htr/curate/jev/scores/part-*.jsonl") for l in open(p)]
    pairs = dict(line.rstrip("\n").split("\t") for line in open("runs/htr/curate/jev/pairs.tsv"))
    rng = random.Random(5); bad = [s for s in scans if s["low"] is not None and s["low"] > 0.4 and s["lines"] >= 10]; mid = [s for s in scans if s["low"] is not None and 0.05 < s["low"] < 0.15 and s["lines"] >= 10]
    picked = rng.sample(bad, 4) + rng.sample(mid, 4)
    jev = LLM(model="runs/htr/final/hub/diffu-minicpm-ocr-htr-jev", runner="pooling", convert="classify", hf_overrides={"num_labels": 2}, limit_mm_per_prompt={"image": 2}, gpu_memory_utilization=0.22)
    rows = []
    for s in picked:
        lines = alto_lines(Path(pairs[s["image"]]))
        with Image.open(s["image"]) as scan:
            scan.load(); crops = [line_crop(scan, l) for l in lines]
        outs = jev.classify([{"prompt": model_prompt(l.text), "multi_modal_data": {"image": [c, render_text(l.text, FONT, c.size)]}} for l, c in zip(lines, crops)], pooling_params=PoolingParams(use_activation=False), use_tqdm=False)
        for l, c, o in zip(lines, crops, outs):
            no, yes = o.outputs.probs[:2]; rows.append((1 / (1 + math.exp(no - yes)), l.text, c, Path(s["image"]).stem))
    rows.sort(key=lambda r: r[0]); low = [r for r in rows if r[0] < 0.5]; high = [r for r in rows if r[0] >= 0.9]
    out = {"low": [item(*r) for r in rng.sample(low, min(20, len(low)))], "high": [item(*r) for r in rng.sample(high, min(10, len(high)))], "lines": len(rows), "scans": len(picked)}
    json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False); print(len(rows), "lines scored;", len(low), "under 0.5;", len(high), "over 0.9")
