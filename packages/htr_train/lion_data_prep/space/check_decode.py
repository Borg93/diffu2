"""Check dspark_hf on a GPU: the reader alone and with its draft read the same lines and regions, compared token for
token, and timed; the first lines are also compared with transformers' own `generate`.

    python check_decode.py READER DRAFT LINES.jsonl REGIONS.jsonl [--lines 20] [--regions 12] [--cache 4096]

READER and DRAFT are Hub ids or folders; the sets are `htr-eval`'s lines files (id, image, text).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import dspark_hf as D

IMAGES = {"downsample_mode": "4x", "max_slice_nums": 9}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reader")
    parser.add_argument("draft")
    parser.add_argument("lines", type=Path)
    parser.add_argument("regions", type=Path)
    parser.add_argument("--lines", dest="n_lines", type=int, default=20)
    parser.add_argument("--regions", dest="n_regions", type=int, default=12)
    parser.add_argument("--cache", type=int, default=4096, help="the reader's cache length (prompt and reading)")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.reader)
    model = AutoModelForImageTextToText.from_pretrained(args.reader, dtype=torch.bfloat16).to("cuda").eval()
    folder = args.draft if Path(args.draft).is_dir() else snapshot_download(args.draft)
    draft = D.DSparkDraft.from_folder(folder, model.get_input_embeddings()).to("cuda", torch.bfloat16).eval()
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids) if isinstance(eos_ids, list) else {int(eos_ids)}

    def inputs(row: dict[str, Any]) -> dict[str, Any]:
        content = [{"type": "image", "image": Image.open(row["image"]).convert("RGB")}]
        content.append({"type": "text", "text": "Text Recognition:"})
        batch = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            processor_kwargs=IMAGES,
        )
        return dict(batch.to("cuda")) | {"downsample_mode": IMAGES["downsample_mode"]}

    def rows(path: Path, n: int) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:n]]

    lines, regions = rows(args.lines, args.n_lines), rows(args.regions, args.n_regions)
    started = time.perf_counter()
    engine = D.FastDSpark(model, draft, args.cache)
    engine.warm(inputs(regions[0]), eos)
    print(f"graphs captured in {time.perf_counter() - started:.1f} s")

    with torch.inference_mode():
        same = 0
        for row in lines[:5]:
            x = inputs(row)
            reference = model.generate(**x, max_new_tokens=512, do_sample=False)[0, x["input_ids"].shape[1] :].tolist()
            same += D.greedy(engine.reader, x, 512, eos).tokens == reference
        print(f"the reader alone on graphs against transformers' generate: {same} of {min(5, len(lines))} identical")

    for name, sample, max_new in (("lines", lines, 512), ("regions", regions, 4096)):
        alone, drafted = [], []
        for row in sample:
            x = inputs(row)
            alone.append(D.greedy(engine.reader, x, max_new, eos))
            drafted.append(engine.decode(x, max_new, eos))
        a, d = sum(r.seconds for r in alone), sum(r.seconds for r in drafted)
        identical = sum(x.tokens == y.tokens for x, y in zip(alone, drafted, strict=True))
        acceptance = statistics.fmean(r.acceptance_length for r in drafted if r.cycles)
        tokens = sum(len(r.tokens) for r in alone)
        print(
            f"{name}: {len(sample)} read, {tokens} tokens; reader alone {a:.2f} s, with the draft {d:.2f} s "
            f"({a / d:.2f}x); identical {identical} of {len(sample)}; acceptance length {acceptance:.2f}"
        )


if __name__ == "__main__":
    main()
