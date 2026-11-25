"""Jev choosing between two readers, line by line: on the 5,000 test_0050 lines, where TrOCR's and our reader's greedy
readings differ, the published Jev scores both (its own 48 px test crop of the line, diffu's prompt and renderer, vLLM's
offline classifier with the card's arguments) and the higher score is kept. Writes the kept readings, the better of
the two by each line's CER (the most a chooser could gain) and each line's scores, in OUT, as htr-eval reads them.

    CUDA_VISIBLE_DEVICES=2 packages/htr_train/envs/vllm/.venv/bin/python runs/htr/verify/jev_pick.py OUT
"""

import json
import math
import sys
from pathlib import Path

from PIL import Image
from vllm import LLM, PoolingParams

from diffu_htr_train.jev.prompt import model_prompt
from diffu_htr_train.jev.render import render_text

JEV = "Gabriel/diffu-minicpm-ocr-htr-jev"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
CROPS = Path("/mnt/work/gabriel/htr/jev_lines.test_crops")  # Jev's test crops: 48 px high, JPEG 95
TROCR = Path("runs/htr/ckpt_eval/test_trocr.jsonl")
OURS = Path("runs/htr/final/test_minicpm.jsonl")


def rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if row}


def same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def distance(a: str, b: str) -> int:
    """Levenshtein distance (this environment has no rapidfuzz)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def cer(prediction: str, truth: str) -> float:
    return distance(prediction, truth) / max(1, len(truth))


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    trocr, ours = rows(TROCR), rows(OURS)
    if trocr.keys() != ours.keys():
        raise SystemExit("the two readers read different lines")
    ids = list(ours)
    differ = [i for i in ids if not same(trocr[i]["prediction"], ours[i]["prediction"])]
    jev = LLM(
        model=JEV, runner="pooling", convert="classify", hf_overrides={"num_labels": 2},
        limit_mm_per_prompt={"image": 2}, gpu_memory_utilization=0.25,
    )
    prompts = []
    for i in differ:
        crop = Image.open(CROPS / f"{i}.jpg").convert("RGB")
        for text in (trocr[i]["prediction"], ours[i]["prediction"]):
            prompts.append({"prompt": model_prompt(text), "multi_modal_data": {"image": [crop, render_text(text, FONT, crop.size)]}})
    outputs = jev.classify(prompts, pooling_params=PoolingParams(use_activation=False))
    scores = [1 / (1 + math.exp(o.outputs.probs[0] - o.outputs.probs[1])) for o in outputs]
    scored = {i: (scores[2 * k], scores[2 * k + 1]) for k, i in enumerate(differ)}

    picked, best, lines = [], [], []
    for i in ids:
        truth, a, b = ours[i]["text"], trocr[i]["prediction"], ours[i]["prediction"]
        score_a, score_b = scored.get(i, (None, None))
        jev_takes = "same" if i not in scored else ("ours" if score_b > score_a else "trocr")
        cer_a, cer_b = cer(a, truth), cer(b, truth)
        picked.append(ours[i] | {"prediction": b if jev_takes == "ours" else a})
        best.append(ours[i] | {"prediction": b if cer_b < cer_a else a})
        lines.append({"id": i, "trocr": a, "ours": b, "trocr_score": score_a, "ours_score": score_b,
                      "trocr_cer": cer_a, "ours_cer": cer_b, "jev_takes": jev_takes})
    for name, items in (("jev_pick", picked), ("best_of_two", best), ("lines", lines)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")

    chosen = [r for r in lines if r["jev_takes"] != "same"]
    better = [r for r in chosen if r["trocr_cer"] != r["ours_cer"]]
    right = [r for r in better if (r["trocr_cer"] < r["ours_cer"]) == (r["jev_takes"] == "trocr")]
    one_exact = [r for r in chosen if (r["trocr_cer"] == 0) != (r["ours_cer"] == 0)]
    exact_right = [r for r in one_exact if (r["trocr_cer"] == 0) == (r["jev_takes"] == "trocr")]
    kept = {"trocr": sum(r["jev_takes"] == "trocr" for r in chosen), "ours": sum(r["jev_takes"] == "ours" for r in chosen)}
    print(f"{len(ids)} lines; the two readings differ on {len(chosen)}; Jev kept TrOCR's on {kept['trocr']}, ours on {kept['ours']}")
    print(f"one reading closer to the truth on {len(better)}: Jev kept the closer one on {len(right)} ({100 * len(right) / max(1, len(better)):.1f} %)")
    print(f"exactly one reading right on {len(one_exact)}: Jev kept the right one on {len(exact_right)} ({100 * len(exact_right) / max(1, len(one_exact)):.1f} %)")
    summary = {"lines": len(ids), "differ": len(chosen), "kept": kept, "one_closer": len(better), "kept_closer": len(right),
               "one_right": len(one_exact), "kept_right": len(exact_right)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
