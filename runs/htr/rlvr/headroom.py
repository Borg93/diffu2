"""How much better than the greedy reading the best of N sampled readings is, by temperature (CURATION E72).

RL can only move the reader towards answers it samples. At temperature 1 the best of 8 samples was worse than the
greedy reading on most pages (pilot 2b's log), so the greedy reading barely moved. This reads each page of a split
greedy (repetition penalty 1.0, as the rollouts) and N times at each temperature named, with the rollouts' engine and
limits (``htr_evaluate.evaluation_protocol``), scores every answer with the reward's parts and writes

    OUT/samples.jsonl   one row a page and temperature: the answers' CER, reward, looped, capped (greedy: temperature 0)
    OUT/summary.json    per temperature: pooled CER of greedy, of the samples' mean, of the best sample (by reward, as
                        RL sees it, and by CER, the ceiling), the share of pages where a sample beats greedy, the
                        spread of the reward within a page, the share of answers looped or capped

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src:runs/htr/rlvr \\
        /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python runs/htr/rlvr/headroom.py VIEW OUT \\
        --temperatures 0.3,0.6,1.0 --samples 8 [--data DIR] [--split monitor] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from eval_read import DATA, clean, pages
from eval_score import page_text
from htr_reward import components
from htr_settings import settings
from nemotron.recipes.parse.document_rewards.baselines import _distance


def scored(answer, row: dict, eos: int, weights: dict, loop: dict) -> dict:
    """One answer's CER and reward parts."""
    raw = clean(answer.text)
    capped = answer.finish_reason == "length"
    ended = bool(answer.token_ids) and answer.token_ids[-1] == eos
    parts = components(raw, row["target"], weights=weights, loop=loop, ended=ended, capped=capped)
    truth = page_text(row["target"])
    return {
        "edits": _distance(truth, page_text(raw)),
        "chars": len(truth),
        "reward": parts["reward"],
        "looped": int(parts["looped"]),
        "capped": int(capped),
        "tokens": len(answer.token_ids),
    }


def summarise(rows: list[dict], greedy: dict[str, dict]) -> dict:
    """The pooled figures of one temperature's rows (``answers`` a page) against the greedy answers."""
    chars = sum(greedy[row["id"]]["chars"] for row in rows)
    count = sum(len(row["answers"]) for row in rows)

    def pooled(pick) -> float:
        return round(100 * sum(pick(row) for row in rows) / chars, 3)

    def by_reward(row: dict) -> dict:
        return max(row["answers"], key=lambda a: a["reward"])

    clean_rows = [
        row for row in rows if not any(a["capped"] or a["looped"] for a in [*row["answers"], greedy[row["id"]]])
    ]
    clean_chars = sum(greedy[row["id"]]["chars"] for row in clean_rows)
    return {
        "pages": len(rows),
        "greedy_cer": pooled(lambda row: greedy[row["id"]]["edits"]),
        "mean_sample_cer": pooled(lambda row: statistics.fmean(a["edits"] for a in row["answers"])),
        "best_by_reward_cer": pooled(lambda row: by_reward(row)["edits"]),
        "best_by_cer": pooled(lambda row: min(a["edits"] for a in row["answers"])),
        "pages_a_sample_beats_greedy": sum(
            min(a["edits"] for a in row["answers"]) < greedy[row["id"]]["edits"] for row in rows
        ),
        "pages_best_by_reward_beats_greedy": sum(by_reward(row)["edits"] < greedy[row["id"]]["edits"] for row in rows),
        "reward_spread_in_page": round(statistics.fmean(statistics.pstdev(a["reward"] for a in row["answers"]) for row in rows), 4),
        "answers_looped": round(sum(a["looped"] for row in rows for a in row["answers"]) / count, 4),
        "answers_capped": round(sum(a["capped"] for row in rows for a in row["answers"]) / count, 4),
        "clean_pages": len(clean_rows),
        "clean_greedy_cer": round(100 * sum(greedy[r["id"]]["edits"] for r in clean_rows) / max(1, clean_chars), 3),
        "clean_best_by_reward_cer": round(100 * sum(by_reward(r)["edits"] for r in clean_rows) / max(1, clean_chars), 3),
        "clean_best_by_cer": round(100 * sum(min(a["edits"] for a in r["answers"]) for r in clean_rows) / max(1, clean_chars), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("view", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--temperatures", default="0.3,0.6,1.0")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--split", default="monitor")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import htr_evaluate
    from PIL import Image
    from vllm import LLM, SamplingParams

    rows = pages(args.data, args.split)[: args.limit]
    configured = settings()
    protocol = htr_evaluate.evaluation_protocol(args.samples, configured, [1.0])
    llm = LLM(model=str(args.view), trust_remote_code=True, **protocol["engine"])
    tokenizer = llm.get_tokenizer()
    requests = []
    for row in rows:
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
        ids = [tokenizer.pad_token_id] + tokenizer.encode(row["prompt"], add_special_tokens=False)
        requests.append({"prompt_token_ids": ids, "multi_modal_data": {"image": image}})

    def read(params: dict) -> list[dict]:
        outputs = llm.generate(requests, SamplingParams(**params))
        return [
            {
                "id": row["id"],
                "temperature": params["temperature"],
                "answers": [
                    scored(a, row, tokenizer.eos_token_id, configured["weights"], configured["loop"]) for a in out.outputs
                ],
            }
            for row, out in zip(rows, outputs, strict=True)
        ]

    args.out.mkdir(parents=True, exist_ok=True)
    first = read(protocol["greedy"][0])
    greedy = {row["id"]: row["answers"][0] for row in first}
    written, summary = list(first), {}
    for temperature in map(float, args.temperatures.split(",")):
        sampled = read({**protocol["generation"], "temperature": temperature})
        written += sampled
        summary[f"{temperature:g}"] = summarise(sampled, greedy)
        print(f"temperature {temperature:g}: {json.dumps(summary[f'{temperature:g}'])}", flush=True)
        (args.out / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in written), "utf-8")
        (args.out / "summary.json").write_text(json.dumps({"view": str(args.view), "samples": args.samples, "by_temperature": summary}, indent=1), "utf-8")


if __name__ == "__main__":
    main()
