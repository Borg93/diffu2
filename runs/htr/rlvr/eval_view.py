"""A model folder one checkpoint is read from by all three readers: the SFT checkpoint's files with its weights.

    /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python runs/htr/rlvr/eval_view.py CHECKPOINT VIEW

The RL stage saves ``checkpoints/step_N/policy/weights/model/consolidated``: weights, a ``config.json`` the recipe
rewrote for its training wrapper (``auto_map`` -> ``hf_parse.ParseForConditionalGeneration``, ``parse_train_scope``)
and the model code, but no tokenizer, processor or chat template. The recipe's own reading of it (``htr_evaluate``,
like ``dental/evaluate.py`` and ``training/model.py``'s ``prepare_model``) builds a view: the checkpoint's files linked,
``architectures`` set, the tokenizer and processor files linked from the SFT checkpoint (``processor_source``).

Here the view is the SFT checkpoint's own folder with the weights swapped: every top-level file of the SFT checkpoint
(config, generation config, code, tokenizer, processor, chat template) linked, except its weights and training state,
and the checkpoint's safetensors linked as ``model.safetensors`` (or its shards with their index). This works because
the RL weights have the SFT weights' names, dtypes and shapes (checked here, tensor by tensor from the headers): RL
trains the same module tree. So the Transformers reader loads the SFT class with its generation config
(repetition penalty 1.1), the server gets the same files as ``/mnt/scratch/gabriel/htr/serve/parse-selftrain2``, and
offline vLLM gets what ``htr_evaluate`` gave it. For the SFT checkpoint itself the view is that checkpoint.

Writes ``VIEW/view.json`` (sources, and the checks made).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SFT = REPO / "runs/htr/nemotron-parse-selftrain-finetune2/output/checkpoint-3000"
NOT_LINKED = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "trainer_state.json"}


def header(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """A safetensors file's tensors: name -> (dtype, shape), from its header only."""
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        table = json.loads(stream.read(size))
    table.pop("__metadata__", None)
    return {name: (entry["dtype"], tuple(entry["shape"])) for name, entry in table.items()}


def tensors(folder: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    found: dict[str, tuple[str, tuple[int, ...]]] = {}
    for path in sorted(folder.glob("*.safetensors")):
        found.update(header(path))
    return found


def build(checkpoint: Path, view: Path, sft: Path = SFT) -> dict:
    """Make ``view`` for ``checkpoint`` (raises when its tensors are not the SFT model's)."""
    checkpoint, sft = checkpoint.resolve(), sft.resolve()
    weights = sorted(checkpoint.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No safetensors in {checkpoint}")
    mine, theirs = tensors(checkpoint), tensors(sft)
    if mine != theirs:
        missing, extra = sorted(set(theirs) - set(mine)), sorted(set(mine) - set(theirs))
        changed = sorted(name for name in set(mine) & set(theirs) if mine[name] != theirs[name])
        raise ValueError(f"Not the SFT model's tensors: missing {missing[:5]}, extra {extra[:5]}, changed {changed[:5]}")
    record = {
        "checkpoint": str(checkpoint),
        "sft": str(sft),
        "tensors": len(mine),
        "tensors_match_sft": True,
        "weights": {},
        "linked_from_sft": [],
    }
    if view.exists():
        saved = json.loads((view / "view.json").read_text())
        if saved["checkpoint"] != record["checkpoint"]:
            raise FileExistsError(f"{view} is a view of {saved['checkpoint']}")
        return saved
    view.mkdir(parents=True)
    for path in sorted(sft.iterdir()):
        if path.is_file() and path.suffix != ".safetensors" and path.name not in NOT_LINKED:
            if path.name == "model.safetensors.index.json":
                continue
            (view / path.name).symlink_to(path)
            record["linked_from_sft"].append(path.name)
    if len(weights) == 1:
        (view / "model.safetensors").symlink_to(weights[0])
        record["weights"]["model.safetensors"] = str(weights[0])
    else:
        index = checkpoint / "model.safetensors.index.json"
        if not index.is_file():
            raise FileNotFoundError(f"Sharded weights without an index: {checkpoint}")
        for path in [*weights, index]:
            (view / path.name).symlink_to(path)
            record["weights"][path.name] = str(path)
    for name, target in record["weights"].items():
        stat = Path(target).stat()
        record["weights"][name] = {"path": target, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    config = json.loads((view / "config.json").read_text())
    record["config_sha256"] = hashlib.sha256((view / "config.json").read_bytes()).hexdigest()
    record["architectures"] = config.get("architectures")
    record["auto_model"] = config.get("auto_map", {}).get("AutoModel")
    needed = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "preprocessor_config.json", "processor_config.json", "chat_template.jinja")
    absent = [name for name in needed if not (view / name).is_file()]
    if absent:
        raise FileNotFoundError(f"The view lacks {absent}")
    (view / "view.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("view", type=Path)
    parser.add_argument("--sft", type=Path, default=SFT)
    args = parser.parse_args()
    record = build(args.checkpoint, args.view, args.sft)
    print(json.dumps({key: record[key] for key in ("checkpoint", "tensors", "tensors_match_sft", "architectures",
                                                   "auto_model")}))


if __name__ == "__main__":
    main()
