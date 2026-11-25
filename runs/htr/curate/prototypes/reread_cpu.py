"""The pool pages the server's repetition stop cut short, read again on the CPU with the guarded reader (CURATION E72).

The pool read serves the reader through vLLM with a repetition stop: a page that starts to loop is cut there
(``finish`` ``repetition``), so its reading is partial. The reader with Transformers' guard (no 10-gram twice) is the
one that reads the GT test pages at 6.63 %; it blocks the loop instead of stopping. This reads every cut page of the
sets named with that reader, on the CPU (bfloat16; the GPU is the pool read's and the RL runs'), shard ``K`` of ``N``,
appending to ``runs/htr/curate/pool_read/reread_cpu/SET.shardK.jsonl`` (resumable: ids written are skipped).

    CUDA_VISIBLE_DEVICES= nice -n 15 packages/htr_train/envs/nemotron_parse/.venv/bin/python \\
        runs/htr/curate/prototypes/reread_cpu.py K N THREADS SET [SET ...]

Rows: ``id``, ``set``, ``raw`` (the answer with its layout tokens), ``prediction`` (plain text, as the pool read
writes it), ``tokens`` (generated), ``finish`` (``length`` at the token cap, else ``stop``), ``seconds``, ``reader``.
"""

import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

CHECKPOINT = "runs/htr/nemotron-parse-selftrain-finetune2/output/checkpoint-3000"
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
SETS = Path("/mnt/scratch/gabriel/htr")
READS = Path("runs/htr/curate/pool_read")
MAX_NEW = 4000
NO_REPEAT = 10
LAYOUT = re.compile(r"<(?:x|y)_[0-9.]+>|<class_[A-Za-z-]+>")
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\| & ")


def text(answer: str) -> str:
    return " ".join(TABLE.sub(" ", LAYOUT.sub("\n", answer)).replace("#", " ").split())


def lines(path: Path) -> list[str]:
    """The file's lines, split at newlines alone (``str.splitlines`` also splits at U+0085 inside a label)."""
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line]


def cut_pages(name: str) -> list[str]:
    """The ids of the set's pages whose served reading did not end by itself, in the read's order."""
    return [row["id"] for row in map(json.loads, lines(READS / f"{name}_vllm.jsonl")) if row.get("finish") != "stop"]


def main() -> None:
    shard, shards, threads, names = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    torch.set_num_threads(threads)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True, dtype=torch.bfloat16).eval()
    out_dir = READS / "reread_cpu"
    out_dir.mkdir(exist_ok=True)
    for name in names:
        rows = lines(SETS / f"{name}_nemotron/pages.jsonl")
        metas = [json.loads(line)["id"] for line in lines(SETS / f"{name}_nemotron/pages.meta.jsonl")]
        if len(rows) != len(metas):
            raise SystemExit(f"{name}: {len(rows)} pages, {len(metas)} lines of meta")
        index = {page: k for k, page in enumerate(metas)}
        out = out_dir / f"{name}.shard{shard}.jsonl"
        done = {json.loads(line)["id"] for line in lines(out)} if out.exists() else set()
        todo = [page for k, page in enumerate(cut_pages(name)) if k % shards == shard and page not in done]
        print(f"{name}: {len(todo)} pages to read in shard {shard} of {shards} ({len(done)} before)", flush=True)
        started = time.time()
        with out.open("a", encoding="utf-8") as sink:
            for count, page in enumerate(todo, 1):
                image = json.loads(rows[index[page]])["images"][0]
                with Image.open(image) as picture:
                    inputs = processor(
                        images=[picture.convert("RGB")], text=PROMPT, return_tensors="pt", add_special_tokens=False
                    )
                began = time.time()
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=MAX_NEW, no_repeat_ngram_size=NO_REPEAT)
                new = ids.shape[1] - inputs["input_ids"].shape[1]
                answer = processor.batch_decode(ids, skip_special_tokens=False)[0].split(PROMPT, 1)[-1]
                raw = answer.replace("</s>", "").replace("<s>", "").replace("<pad>", "")
                row = {
                    "id": page,
                    "set": name,
                    "raw": raw,
                    "prediction": text(raw),
                    "tokens": int(new),
                    "finish": "length" if new >= MAX_NEW else "stop",
                    "seconds": round(time.time() - began, 1),
                    "reader": f"{CHECKPOINT} transformers bf16 cpu no_repeat_ngram_size={NO_REPEAT}",
                }
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                if count % 25 == 0:
                    print(f"{name}: {count} of {len(todo)}, {count / (time.time() - started) * 60:.1f} a minute", flush=True)


if __name__ == "__main__":
    main()
