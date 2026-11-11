"""ms-swift's encoding of a set's samples against the model's own chat template, for the envs lane: run by a tool
environment's own Python with a rendered command's template flags and environment, never imported by the tests.

    <env>/.venv/bin/python envs_template.py MODEL SET --model_type T --template T --add_non_thinking_prefix B \\
        --loss_scale S

A JSON line a sample: whether ms-swift's token ids are those of the model's ``chat_template.jinja`` (the
conversation with its answer, thinking off), whether its image grid is the processor's own, and whether the loss starts
where the template's generation prompt ends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from swift import get_processor, get_template
from transformers import AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("set", type=Path)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--add_non_thinking_prefix", required=True, choices=["true", "false"])
    parser.add_argument("--loss_scale", required=True)
    args = parser.parse_args()
    processor = get_processor(str(args.model), model_type=args.model_type, use_hf=True, download_model=False)
    template = get_template(
        processor,
        template_type=args.template,
        add_non_thinking_prefix=args.add_non_thinking_prefix == "true",
        loss_scale=args.loss_scale,
    )
    template.set_mode("train")
    own = AutoProcessor.from_pretrained(str(args.model))
    for line in args.set.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        asked, answer = row["messages"][0]["content"].removeprefix("<image>"), row["messages"][1]["content"]
        content = [{"type": "image", "image": Image.open(row["images"][0]).convert("RGB")}]
        user = {"role": "user", "content": content + ([{"type": "text", "text": asked}] if asked else [])}
        conversation = [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        tokens = {"tokenize": True, "return_dict": True, "return_tensors": "pt", "enable_thinking": False}
        whole = own.apply_chat_template(conversation, **tokens)
        prompt = own.apply_chat_template([user], add_generation_prompt=True, **tokens)
        encoded = template.encode(row)
        first = next(k for k, label in enumerate(encoded["labels"]) if label != -100)
        result = {
            "same_ids": list(encoded["input_ids"]) == whole["input_ids"][0].tolist(),
            "same_grid": encoded["image_grid_thw"].tolist() == whole["image_grid_thw"].tolist(),
            "loss_from_generation_prompt_end": first == prompt["input_ids"].shape[1],
        }
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
