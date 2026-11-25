"""Read a split's pages with one checkpoint (a view, ``eval_view.py``) by one of three readers; raw readings only.

    # (1) offline vLLM, greedy, as htr_evaluate reads (parse_rlvr environment; one file per penalty)
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src:runs/htr/rlvr \\
        /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python runs/htr/rlvr/eval_read.py vllm VIEW SPLIT OUT_DIR
    # (2) Transformers, greedy with the checkpoint's generation config (repetition penalty 1.1) and no 10-gram twice,
    #     at most 4,000 new tokens: runs/htr/verify/nemotron_parse_test_read.py with NO_REPEAT_NGRAM=10 and
    #     checkpoint_cer.read (the same decode) (envs/nemotron_parse environment)
    CUDA_VISIBLE_DEVICES=2 packages/htr_train/envs/nemotron_parse/.venv/bin/python runs/htr/rlvr/eval_read.py \\
        transformers VIEW SPLIT OUT_DIR
    # (3) a vLLM server (runs/htr/parse_serve.sh VIEW) with vLLM's repetition stop: runs/htr/verify/parse_vllm_read.py's
    #     request (temperature 0, top-k 1, repetition penalty 1.1, 8,744 tokens, repetition_detection 4..120 x 6)
    URL=http://127.0.0.1:8021/v1 .../python runs/htr/rlvr/eval_read.py served VIEW SPLIT OUT_DIR

SPLIT is ``test`` (the 197 test pages) or ``monitor`` (the 115 dev pages: 5 whole dev volumes), read from the dataset
bundle (``--data``, data_v2 by default: the same pages as v1), whose manifests are read on "\\n" only. The reference
scripts read the 200 test pages by the order of other files; this reads the bundle's pages, so all readers and
checkpoints read the same pages, and the scorer (``eval_score.py``) needs no other file.

Writes ``OUT_DIR/<reader>.<split>.jsonl``: ``id``, ``raw`` (special tokens kept, </s> <s> <pad> removed), ``tokens``
(new tokens), ``finish`` (stop / length / the server's reason), ``ended`` (the last token was EOS; None when unknown),
``seconds``. A file that is complete is not read again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DATA = Path("/mnt/scratch/gabriel/htr/rlvr/data_v2")
MAX_NEW = 4000  # nemotron_parse_test_read.py and checkpoint_cer.py
NO_REPEAT = 10  # the reads named norepeat10
SERVED_MAX = 8744  # parse_vllm_read.py
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def pages(data: Path, split: str) -> list[dict]:
    """The split's manifest rows, images absolute."""
    rows = [json.loads(line) for line in (data / f"{split}.jsonl").read_text("utf-8").split("\n") if line.strip()]
    for row in rows:
        row["image"] = str((data / row["image"]).resolve())
        if row["prompt"] != PROMPT:
            raise ValueError(f"Unexpected prompt on {row['id']}")
    return rows


def clean(raw: str) -> str:
    return raw.replace("</s>", "").replace("<s>", "").replace("<pad>", "")


def write(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8")
    temporary.replace(path)


def done(path: Path, count: int) -> bool:
    return path.is_file() and len([l for l in path.read_text("utf-8").split("\n") if l.strip()]) == count


def read_vllm(view: Path, rows: list[dict], out: Path, split: str, penalties: list[float]) -> None:
    """Greedy readings at each repetition penalty with htr_evaluate's engine and limits (its protocol)."""
    import htr_evaluate
    from htr_settings import settings
    from PIL import Image
    from vllm import LLM, SamplingParams

    targets = {p: out / f"vllm_p{p:g}.{split}.jsonl" for p in penalties}
    if all(done(path, len(rows)) for path in targets.values()):
        return
    protocol = htr_evaluate.evaluation_protocol(1, settings(), penalties)
    llm = LLM(model=str(view), trust_remote_code=True, **protocol["engine"])
    tokenizer = llm.get_tokenizer()
    requests = []
    for row in rows:
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
        ids = [tokenizer.pad_token_id] + tokenizer.encode(row["prompt"], add_special_tokens=False)
        requests.append({"prompt_token_ids": ids, "multi_modal_data": {"image": image}})
    for params in protocol["greedy"]:
        path = targets[params["repetition_penalty"]]
        started = time.perf_counter()
        outputs = llm.generate(requests, SamplingParams(**params))
        seconds = round((time.perf_counter() - started) / len(rows), 3)
        write(
            path,
            [
                {
                    "id": row["id"],
                    "raw": clean(answer.outputs[0].text),
                    "tokens": len(answer.outputs[0].token_ids),
                    "finish": answer.outputs[0].finish_reason,
                    "ended": bool(answer.outputs[0].token_ids)
                    and answer.outputs[0].token_ids[-1] == tokenizer.eos_token_id,
                    "seconds": seconds,  # the batch's time over its pages
                    "max_tokens": params["max_tokens"],
                }
                for row, answer in zip(rows, outputs, strict=True)
            ],
        )


def read_transformers(view: Path, rows: list[dict], out: Path, split: str) -> None:
    """One page at a time, as nemotron_parse_test_read.py (NO_REPEAT_NGRAM=10) and checkpoint_cer.read decode."""
    path = out / f"hf_norepeat10.{split}.jsonl"
    if done(path, len(rows)):
        return
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(view, trust_remote_code=True)
    model = AutoModel.from_pretrained(view, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    eos = processor.tokenizer.eos_token_id
    results = []
    for row in rows:
        with Image.open(row["image"]) as image:
            inputs = processor(
                images=[image.convert("RGB")], text=row["prompt"], return_tensors="pt", add_special_tokens=False
            ).to("cuda")
        started = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=MAX_NEW, no_repeat_ngram_size=NO_REPEAT)
        answer = processor.batch_decode(ids, skip_special_tokens=False)[0].split(row["prompt"], 1)[-1]
        new = int(ids.shape[1]) - int(inputs["input_ids"].shape[1])
        # The generation config has forced_eos_token_id: a reading cut at max_new_tokens also ends with EOS, so the
        # cap is told by the count (checked on CPU, 25 Nov: a 40-token read ends in </s>).
        capped = new >= MAX_NEW
        results.append(
            {
                "id": row["id"],
                "raw": clean(answer),
                "tokens": new,
                "finish": "length" if capped else "stop",
                "ended": int(ids[0, -1]) == eos and not capped,
                "seconds": round(time.perf_counter() - started, 2),
                "max_tokens": MAX_NEW,
            }
        )
    write(path, results)


def read_served(rows: list[dict], out: Path, split: str, concurrency: int) -> None:
    """parse_vllm_read.py's request against the server at URL (MODEL: its served name)."""
    import base64
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    path = out / f"served_stop.{split}.jsonl"
    if done(path, len(rows)):
        return
    url, name = os.environ.get("URL", "http://127.0.0.1:8021/v1"), os.environ.get("MODEL", "nemotron-parse")
    images = {row["id"]: base64.b64encode(Path(row["image"]).read_bytes()).decode() for row in rows}
    client = httpx.Client(timeout=1800, limits=httpx.Limits(max_connections=max(64, concurrency)))

    def read(row: dict) -> dict:
        started = time.perf_counter()
        body = {
            "model": name, "temperature": 0, "max_tokens": SERVED_MAX, "top_k": 1, "repetition_penalty": 1.1,
            "skip_special_tokens": False,
            "repetition_detection": {"min_pattern_size": 4, "max_pattern_size": 120, "min_count": 6},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + images[row["id"]]}},
            ]}],
        }
        answer = client.post(f"{url}/chat/completions", json=body).raise_for_status().json()
        choice = answer["choices"][0]
        return {
            "id": row["id"],
            "raw": clean(choice["message"]["content"] or ""),
            "tokens": answer["usage"]["completion_tokens"],
            "finish": choice["finish_reason"],
            "ended": None,  # the server does not say whether the last token was EOS; finish says why it stopped
            "seconds": round(time.perf_counter() - started, 2),
            "max_tokens": SERVED_MAX,
        }

    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(read, rows))
    write(path, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("reader", choices=("vllm", "transformers", "served"))
    parser.add_argument("view", type=Path)
    parser.add_argument("split", choices=("test", "monitor"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--penalties", default="1.0,1.1", help="vllm: the greedy repetition penalties")
    parser.add_argument("--concurrency", type=int, default=32, help="served: requests in flight")
    parser.add_argument("--limit", type=int, help="the first N pages only (a quick check)")
    args = parser.parse_args()
    rows = pages(args.data, args.split)[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.reader == "vllm":
        read_vllm(args.view, rows, args.out, args.split, [float(p) for p in args.penalties.split(",")])
    elif args.reader == "transformers":
        read_transformers(args.view, rows, args.out, args.split)
    else:
        read_served(rows, args.out, args.split, args.concurrency)
    print(f"{args.reader} {args.split}: {len(rows)} pages in {time.perf_counter() - started:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
