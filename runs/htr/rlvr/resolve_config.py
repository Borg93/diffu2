"""Resolve the stage and training configuration of a run without starting it (no GPU): ``DRY=1 run_rlvr.sh NAME``.

Does what ``stage2_rlvr/train.py`` does before it touches the GPU: loads the stage configuration, the task, checks
the dataset's identity and the SFT record's lineage, resolves the training YAML with the recipe's ``rlvr_config``
and validates it with NeMo RL's ``MasterConfig``. Prints the settings that differ from the recipe's ``grpo.yaml``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def flat(value, prefix=""):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from flat(inner, f"{prefix}{key}.")
    else:
        yield prefix[:-1], value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-identity", action="store_true", help="do not re-hash the dataset and the models")
    args = parser.parse_args()

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemotron.recipes.parse.common import validate_dataset
    from nemotron.recipes.parse.config import ROOT, load_config
    from nemotron.recipes.parse.training.config import rlvr_config
    from nemotron.recipes.parse.training.lineage import read_lineage
    from nemotron.recipes.parse.training.task import load_task

    cfg = load_config("rlvr", args.stage, [f"output_dir={args.out}", f"training_config={args.training}"])
    print("stage:", json.dumps(cfg.model_dump(mode="json"), indent=2))
    task = load_task(cfg.task, stage="rlvr")
    print("task:", cfg.task, "| environment:", task.environment_class, "| reward metric:", task.reward_metric)
    from htr_settings import data_dir, settings

    if data_dir() != Path(cfg.data_dir).resolve():
        raise ValueError(f"The task's hooks would draw from {data_dir()}, the stage validates {cfg.data_dir}")
    hooks = settings()
    print("task hooks read:", hooks["config"], "| data:", data_dir())
    print("  candidates:", json.dumps(hooks["candidates"]))
    print("  selection shares: failing", hooks["failing_share"], "informed", hooks["informed_share"],
          "| rollout temperature", hooks["temperature"], "| greedy penalties", hooks["greedy_repetition_penalties"])
    if not args.skip_identity:
        identity = validate_dataset(cfg.data_dir)
        print("dataset identity holds:", identity["pages"], identity["contexts"])
        lineage = read_lineage(cfg.sft_output / "artifact.json", "sft", identity)
        print("SFT lineage holds:", {name: ref["path"] for name, ref in lineage["models"].items()})
    out = Path(cfg.output_dir)
    model, curriculum = out / "initial-model", out / "curriculum"

    def resolved(path, overrides):
        return rlvr_config(path, model, curriculum, out, prompt_file=task.prompt_file, overrides=overrides)

    ours = resolved(cfg.training_config, cfg.training)
    MasterConfig(**ours)
    print("training configuration validates as NeMo RL's MasterConfig")
    theirs = dict(flat(resolved(ROOT / "stage2_rlvr/config/grpo.yaml", {})))
    print("settings that differ from the recipe's grpo.yaml (ours <- theirs):")
    for key, value in flat(ours):
        if theirs.get(key, "<unset>") != value:
            print(f"  {key}: {value!r} <- {theirs.get(key, '<unset>')!r}")


if __name__ == "__main__":
    main()
