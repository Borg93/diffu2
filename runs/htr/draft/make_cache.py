"""A DFlash2 training cache for a Nemotron Parse reader: the reader's hidden states over a page and its own answer.

NeMo AutoModel's cached DFlash route (``components/speculative/dflash/cached_data.py``) reads a folder with a
``manifest.json`` and a ``*.ckpt`` a sample: ``input_ids`` [seq], ``loss_mask`` [seq] and ``hidden_states``
[seq, layers * hidden]. This writes that folder for a Parse reader as vLLM serves it with a DFlash2 draft:

- the sequence is the decoder's whole sequence, the image prefix first and the text after (vLLM numbers the draft's
  positions over both, and the draft's context keys and values come from the reader's states at every position);
- ``input_ids`` at the image positions is 1 (``<pad>``, the placeholder vLLM puts there); they are never anchors;
- ``hidden_states`` is the output of decoder layers ``--layers`` (zero-based, before the final norm), side by side:
  vLLM takes aux state ``i + 1`` for layer id ``i``, and Parse's state ``k + 1`` is what layer ``k`` returns;
- ``loss_mask`` is 1 on the answer's tokens (the end token with them) and 0 on the image and the prompt.

The answer is the reader's own: ``--answers`` (JSON lines with ``id`` and ``raw``, a served reader's readings) or,
with ``--generate``, read here greedily. A ``stats.jsonl`` beside the rows says, a page, how many of the answer's
tokens the reader picks again when it is made to follow the answer (near 1.0 for an answer that is its own).

It also writes the "target tables" folder the trainer reads in place of the reader (``--tables``): a Qwen3-shaped
``config.json`` its registry accepts, the tokenizer, and a ``model.safetensors`` with ``draft_input_embedding.weight``
(the tied table times sqrt(d_model), as Parse's own embedding module returns it and as vLLM hands it to the draft)
and ``lm_head.weight`` (the tied table itself). The manifest names that folder as the target.

Run with ``packages/htr_train/envs/nemotron_parse/.venv/bin/python`` and ``CUDA_VISIBLE_DEVICES=2``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Annotated

import typer

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
IMAGE_TOKEN = 1  # <pad>: what vLLM's Parse processor repeats over the image prefix
EMBEDDING_KEY = "draft_input_embedding.weight"
HEAD_KEY = "lm_head.weight"


def write_tables(reader: Path, tables: Path) -> None:
    """The folder the trainer loads for the target: a config its registry takes, the tokenizer, the two tables."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((reader / "config.json").read_text("utf-8"))
    decoder = config["decoder"]
    tables.mkdir(parents=True, exist_ok=True)
    with safe_open(reader / "model.safetensors", "pt") as weights:
        table = weights.get_tensor("decoder.embed_tokens.weight").float()
        if "lm_head.weight" in weights.keys():  # noqa: SIM118  a safetensors handle, not a dict
            head = weights.get_tensor("lm_head.weight").float()
            if not torch.equal(head, table):
                raise ValueError("the reader's head is not tied to its embedding: the tables must be told apart")
    scale = math.sqrt(decoder["d_model"]) if decoder["scale_embedding"] else 1.0
    save_file({EMBEDDING_KEY: table * scale, HEAD_KEY: table}, tables / "model.safetensors")
    # the fields the trainer copies into the draft's config are NVIDIA's draft's (dflash2/config.json)
    target = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": decoder["vocab_size"],
        "hidden_size": decoder["d_model"],
        "num_hidden_layers": decoder["decoder_layers"],
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 4096,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-06,
        "max_position_embeddings": 16384,
        "initializer_range": 0.02,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_theta": 1000000.0,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "parse_reader": str(reader),
        "parse_embedding_scale": scale,
    }
    (tables / "config.json").write_text(json.dumps(target, indent=2) + "\n", "utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        link = tables / name
        if not link.exists():
            link.symlink_to((reader / name).resolve())


def main(
    reader: Annotated[Path, typer.Option(help="the reader's checkpoint folder")],
    out: Annotated[Path, typer.Option(help="the cache folder to write")],
    tables: Annotated[Path, typer.Option(help="the target tables folder (written if it has no weights)")],
    pages: Annotated[Path, typer.Option(help="the page set: JSON lines with images[0]; ids in its .meta.jsonl")] = Path(
        "/mnt/scratch/gabriel/htr/parse_bench/pages.jsonl"
    ),
    answers: Annotated[Path | None, typer.Option(help="the reader's readings: JSON lines with id and raw")] = None,
    generate: Annotated[bool, typer.Option(help="read the pages here instead (greedy)")] = False,
    repetition_penalty: Annotated[float, typer.Option(help="for --generate")] = 1.1,
    max_new: Annotated[int, typer.Option(help="for --generate: tokens at most")] = 4000,
    start: Annotated[int, typer.Option(help="first page of the set to take")] = 0,
    limit: Annotated[int, typer.Option(help="pages to take")] = 50,
    layers: Annotated[str, typer.Option(help="decoder layers whose output is kept")] = "1,4,8",
    block_size: Annotated[int, typer.Option(help="the draft's block: for the manifest")] = 33,
    mask_token_id: Annotated[int, typer.Option(help="the draft's mask token: for the manifest")] = 3,
    max_seq: Annotated[int, typer.Option(help="skip a page whose sequence is longer")] = 16384,
) -> None:
    """Write the cache of ``limit`` pages of ``pages`` read by ``reader``."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    if (answers is None) == (not generate):
        raise typer.BadParameter("give --answers or --generate, one of them")
    layer_ids = [int(layer) for layer in layers.split(",")]
    if not (tables / "model.safetensors").exists():
        write_tables(reader, tables)
    elif json.loads((tables / "config.json").read_text("utf-8"))["parse_reader"] != str(reader):
        raise typer.BadParameter(f"{tables} holds another reader's tables")

    metas = pages.with_name(pages.name.replace(".jsonl", ".meta.jsonl")).read_text("utf-8").split("\n")
    rows = pages.read_text("utf-8").split("\n")  # not splitlines(): a label may hold U+0085
    metas, rows = [line for line in metas if line], [line for line in rows if line]
    taken = [
        (json.loads(meta)["id"], json.loads(row)["images"][0])
        for row, meta in list(zip(rows, metas, strict=True))[start : start + limit]
    ]
    read = {}
    if answers is not None:
        read = {row["id"]: row for row in map(json.loads, filter(None, answers.read_text("utf-8").split("\n")))}

    processor = AutoProcessor.from_pretrained(reader, trust_remote_code=True)
    model = AutoModel.from_pretrained(reader, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    eos = model.config.eos_token_id
    kept: dict[int, torch.Tensor] = {}
    for layer in layer_ids:
        model.decoder.layers[layer].register_forward_hook(
            lambda _module, _inputs, output, layer=layer: kept.__setitem__(layer, output)
        )

    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        (out / "stats.jsonl").open("w", encoding="utf-8") as stats,
        (out / "answers.jsonl").open("w", encoding="utf-8") as said,
    ):
        for number, (page_id, image_path) in enumerate(taken):
            began = time.perf_counter()
            path = out / f"{start + number:06d}_{page_id}.ckpt"
            if path.exists():
                written += 1
                continue
            if answers is not None and page_id not in read:
                continue
            with Image.open(image_path) as image:
                inputs = processor(
                    images=[image.convert("RGB")], text=PROMPT, return_tensors="pt", add_special_tokens=False
                ).to("cuda")
            prompt_ids = inputs["input_ids"][0]
            with torch.inference_mode():
                if generate:
                    ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=False,
                        num_beams=1,
                        repetition_penalty=repetition_penalty,
                    )[0]
                    answer_ids = ids[len(prompt_ids) :]
                    finish = "stop" if len(answer_ids) and answer_ids[-1] == eos else "length"
                else:
                    row = read[page_id]
                    answer = processor.tokenizer(row["raw"], add_special_tokens=False, return_tensors="pt")
                    answer_ids = answer["input_ids"][0].to("cuda")
                    finish = row.get("finish", "stop")
                    if finish == "stop":
                        answer_ids = torch.cat([answer_ids, answer_ids.new_tensor([eos])])
                text_ids = torch.cat([prompt_ids, answer_ids])
                kept.clear()
                logits = model(
                    input_ids=text_ids[None],
                    pixel_values=inputs["pixel_values"],
                    image_sizes=inputs["image_sizes"],
                    content_sizes=inputs["content_sizes"],
                    use_cache=False,
                ).logits[0]
            states = torch.cat([kept[layer][0] for layer in layer_ids], dim=-1)
            image_len = states.shape[0] - len(text_ids)
            if states.shape[0] > max_seq or len(answer_ids) == 0:
                continue
            again = logits[len(prompt_ids) - 1 : -1].argmax(-1) == answer_ids
            record = {
                "input_ids": torch.cat([text_ids.new_full((image_len,), IMAGE_TOKEN), text_ids]).long().cpu(),
                "loss_mask": torch.cat(
                    [text_ids.new_zeros(image_len + len(prompt_ids)), text_ids.new_ones(len(answer_ids))]
                )
                .long()
                .cpu(),
                "hidden_states": states.to(torch.bfloat16).cpu(),
            }
            torch.save(record, path.with_suffix(".part"))
            path.with_suffix(".part").rename(path)
            written += 1
            raw = processor.tokenizer.decode(answer_ids, skip_special_tokens=False)
            said.write(json.dumps({"id": page_id, "raw": raw, "finish": finish}, ensure_ascii=False) + "\n")
            line = {
                "id": page_id,
                "image_tokens": image_len,
                "prompt_tokens": len(prompt_ids),
                "answer_tokens": len(answer_ids),
                "served_tokens": read.get(page_id, {}).get("tokens"),
                "finish": finish,
                "reader_picks_again": round(again.float().mean().item(), 4),
                "megabytes": round(path.stat().st_size / 1e6, 2),
                "seconds": round(time.perf_counter() - began, 2),
            }
            stats.write(json.dumps(line) + "\n")
            stats.flush()
            if written % 10 == 0:
                print(f"{written} rows, last {line}", flush=True)

    manifest = {
        "format_version": 1,
        "target_model_name_or_path": str(tables),
        "target_layer_ids": layer_ids,
        "target_hidden_size": model.config.decoder.d_model,
        "target_vocab_size": model.config.decoder.vocab_size,
        "block_size": block_size,
        "mask_token_id": mask_token_id,
        "hidden_states_dtype": "bfloat16",
        "visual_score_scale": None,
        "num_samples": len(list(out.glob("*.ckpt"))),
        "reader": str(reader),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    print(f"{written} rows in {out}")


if __name__ == "__main__":
    typer.run(main)
