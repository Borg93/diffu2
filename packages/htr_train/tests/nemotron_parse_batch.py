"""Each page's loss alone and in one batch with the others, for the envs lane (``test_nemotron_parse_envs.py``): run by
``envs/nemotron_parse``'s Python, never imported by the tests.

    envs/nemotron_parse/.venv/bin/python -I nemotron_parse_batch.py SET.jsonl

The set's pages are read and batched as the trainer reads them (:mod:`diffu_htr_train.nemotron_parse.data`), the model
(at its pinned revision, float32, on CPU) reads them all at once and each alone, and the loss of each is printed both
ways, as one JSON line: pages of other sizes give the batch a visual prefix padded between a page's patches and its
text, and answers of other lengths a padded end, which only the masks keep out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoProcessor

from diffu_htr_train.nemotron_parse import REPO, REVISION, TASK_PROMPT
from diffu_htr_train.nemotron_parse.data import PageCollator, page_dataset
from diffu_htr_train.nemotron_parse.loss import LayoutWeightedLoss, layout_mask


def main(path: Path) -> None:
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True, dtype=torch.float32).eval()
    tokenizer = processor.tokenizer
    pages, _ = page_dataset([path], tokenizer, max_tokens=8744)
    prompt = tokenizer(TASK_PROMPT, add_special_tokens=False)["input_ids"]
    collate = PageCollator(processor.image_processor, prompt, tokenizer.pad_token_id)
    loss = LayoutWeightedLoss(layout_mask(tokenizer, model.config.vocab_size), 10.0)

    def losses(batch: dict) -> list[float]:
        labels = batch.pop("labels")
        with torch.no_grad():
            logits = model(**batch).logits
        return [
            loss(SimpleNamespace(logits=logits[row : row + 1]), labels[row : row + 1]).item()
            for row in range(len(labels))
        ]

    items = [pages[index] for index in range(len(pages))]
    batched = losses(collate(items))
    alone = [losses(collate([item]))[0] for item in items]
    sizes = [list(item["image"].size) for item in items]
    print(json.dumps({"alone": alone, "batched": batched, "sizes": sizes}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
