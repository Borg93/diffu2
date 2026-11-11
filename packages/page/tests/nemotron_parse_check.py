"""The model's own code reading a page's answer back, for the envs lane (``test_nemotron_parse.py``): run by
``packages/htr_train/envs/nemotron_parse``'s Python, never imported by the tests.

    envs/nemotron_parse/.venv/bin/python -I nemotron_parse_check.py REQUEST.json

``REQUEST.json`` holds the model's repo and revision, an answer, its page's size (W, H), the coordinates diffu-page
writes and images; printed as one JSON line: those coordinates that are no token of the model's tokenizer and how many x
and y coordinates it has, the answer tokenized as its inference tokenizes text and decoded back, the blocks its
postprocessing reads (header inference off: our tables have no header cells to infer), and the content size (H, W) its
processor makes of each image. Only the model's code, tokenizer and processor files are fetched, never its weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor

FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "hf_*.py",
    "postprocessing.py",
    "latex2html.py",
    "nemotron_parse/*",
]


def main(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    snapshot = Path(snapshot_download(request["repo"], revision=request["revision"], allow_patterns=FILES))
    sys.path.insert(0, str(snapshot))  # its README's own way to import its postprocessing: from the snapshot
    from nemotron_parse.postprocessing import (  # ty: ignore[import-not-found]
        PostprocessOptions,
        postprocess_document,
    )

    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True)
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    not_tokens = [f"<x_{value}>" for value in request["x_coordinates"] if f"<x_{value}>" not in vocab] + [
        f"<y_{value}>" for value in request["y_coordinates"] if f"<y_{value}>" not in vocab
    ]
    ids = tokenizer(request["answer"], add_special_tokens=False)["input_ids"]
    document = postprocess_document(
        request["answer"], image_size=tuple(request["size"]), options=PostprocessOptions(infer_headers=False)
    )
    sizes = []
    for path in request["images"]:
        with Image.open(path) as image:
            sizes.append(processor.image_processor([image.convert("RGB")])["content_sizes"][0].tolist())
    print(
        json.dumps(
            {
                "coordinates_not_tokens": not_tokens,
                "model_coordinates": {
                    "x": sum(token.startswith("<x_") for token in vocab),
                    "y": sum(token.startswith("<y_") for token in vocab),
                },
                "decoded": tokenizer.decode(ids, skip_special_tokens=False),
                "blocks": document["blocks"],
                "content_sizes": sizes,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
