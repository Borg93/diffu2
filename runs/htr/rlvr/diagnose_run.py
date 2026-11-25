"""A GRPO run's training signals at its start, middle and end, read from what the run wrote (CPU, no GPU).

    /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python runs/htr/rlvr/diagnose_run.py \\
        /mnt/scratch/gabriel/htr/rlvr/runs/pilot_1124_2254 [--window 16] [--json OUT.json]

Reads ``history.jsonl`` (the recipe's ``Progress``: NeMo RL's per-update metrics), ``logs/train_data_step*.jsonl``
(NeMo RL's dump of each update's 64 rows: rewards, token advantages, loss masks; the rows of one prompt are
consecutive, ``repeat_interleave``) and ``train-components.jsonl`` (our environment's reward parts per answer).

What each number is:

* ``kl_penalty``: the k3 KL to the reference (SFT) policy as it enters the loss, times nothing (NeMo RL logs the mean).
* ``gen_kl_error`` / ``token_mult_prob_error``: how far vLLM's sampling probabilities are from the trainer's (1.0 =
  equal); not a KL to the reference.
* ``approx_entropy``: NeMo RL's per-token entropy estimate of the sampled answers.
* ``clip``: NeMo RL logs no clip fraction; it logs the ratio's range before and after clamping. With one optimizer
  step per rollout batch (global batch 64 = 16 prompts x 4), the ratio is the trainer against itself on the same
  weights, so its spread is numerical, and clipping acts only on that noise.
* zero advantage: a prompt whose 4 answers got the same reward (|max - min| < 1e-9) has advantage 0 for all 4 and
  teaches nothing; ``near_zero_spread`` counts prompts whose reward std is below 0.01, which ``normalize_rewards``
  still scales up to unit size.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HISTORY = (
    "reward",
    "kl_penalty",
    "grad_norm",
    "approx_entropy",
    "loss",
    "lr",
    "probs_ratio_min",
    "probs_ratio_max",
    "probs_ratio_clamped_min",
    "probs_ratio_clamped_max",
    "token_mult_prob_error",
    "gen_kl_error",
    "sampling_importance_ratio",
    "mean_gen_tokens_per_sample",
    "max_gen_tokens_per_sample",
    "truncation_rate",
    "natural_termination_rate",
    "looped",
    "cer",
    "boxes",
    "predicted_boxes",
    "baseline_reward/pct_0",
    "baseline_reward/pct_1",
    "baseline_reward/pct_mixed",
    "advantages/min",
    "advantages/max",
)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def step_groups(path: Path, generations: int) -> dict:
    """One update's prompts: their reward spread, and whether all their answers got the same advantage of zero."""
    rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    groups = [rows[i : i + generations] for i in range(0, len(rows), generations)]
    flat = zero = near = 0
    spreads, tokens, gen_tokens = [], [], []
    for group in groups:
        rewards = [row["rewards"][0] for row in group]
        if max(rewards) - min(rewards) < 1e-9:
            flat += 1
        std = statistics.pstdev(rewards)
        spreads.append(std)
        near += std < 0.01
        advantages = []
        for row in group:
            mask = row["token_loss_mask"][0]
            values = [a for a, m in zip(row["advantages"][0], mask, strict=True) if m]
            advantages.append(max((abs(a) for a in values), default=0.0))
            gen_tokens.append(sum(mask))
        zero += all(a < 1e-6 for a in advantages)
        tokens.extend(row["input_lengths"][0] for row in group)
    return {
        "prompts": len(groups),
        "flat_reward_prompts": flat,
        "zero_advantage_prompts": zero,
        "near_zero_spread_prompts": near,
        "mean_reward_std": mean(spreads),
        "mean_answer_tokens": mean(gen_tokens),
        "max_answer_tokens": max(gen_tokens),
    }


def window(values: dict[int, dict], steps: list[int]) -> dict:
    """The means over ``steps`` of every key the steps have."""
    keys = sorted({key for step in steps for key in values.get(step, {})})
    return {key: mean([values[step].get(key) for step in steps if step in values]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, default=16, help="updates averaged at the start, middle and end")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    history = [json.loads(line) for line in (args.run / "history.jsonl").read_text().split("\n") if line.strip()]
    train = {event["step"]: event["metrics"] for event in history if event["phase"] == "train"}
    validation = {event["step"]: event["metrics"] for event in history if event["phase"] == "validation"}
    last = max(train)
    middle = last // 2 - args.window // 2
    windows = {
        "start": list(range(1, args.window + 1)),
        "middle": list(range(middle + 1, middle + args.window + 1)),
        "end": list(range(last - args.window + 1, last + 1)),
    }
    groups = {}
    for step in train:
        path = args.run / "logs" / f"train_data_step{step}.jsonl"
        if path.is_file():
            groups[step] = step_groups(path, args.generations)
    report = {"run": str(args.run), "updates": last, "windows": {}}
    for name, steps in windows.items():
        report["windows"][name] = {
            "updates": f"{steps[0]}-{steps[-1]}",
            "history": window({s: {h: train[s].get(h) for h in HISTORY} for s in steps if s in train}, steps),
            "prompts": window(groups, steps),
        }
    totals = {key: sum(g[key] for g in groups.values()) for key in ("prompts", "flat_reward_prompts",
                                                                      "zero_advantage_prompts",
                                                                      "near_zero_spread_prompts")}
    report["all_updates"] = dict(
        totals,
        zero_advantage_share=totals["zero_advantage_prompts"] / max(1, totals["prompts"]),
        near_zero_spread_share=totals["near_zero_spread_prompts"] / max(1, totals["prompts"]),
        grad_norm_max=max(train[s].get("grad_norm", 0) for s in train),
        grad_norm_max_update=max(train, key=lambda s: train[s].get("grad_norm", 0)),
        kl_penalty_max=max(train[s].get("kl_penalty", 0) for s in train),
        truncated_updates=sum(train[s].get("truncation_rate", 0) > 0 for s in train),
        lr_values=sorted({train[s].get("lr") for s in train}),
    )
    report["validation"] = validation
    components = args.run / "train-components.jsonl"
    if components.is_file():
        events = [json.loads(line) for line in components.read_text().split("\n") if line.strip()]
        report["environment_calls"] = len(events)
        report["answers_looped"] = sum(s["looped"] for e in events for s in e["samples"])
        report["answers"] = sum(len(e["samples"]) for e in events)
    text = json.dumps(report, indent=1)
    if args.json:
        args.json.write_text(text + "\n")
    for name, part in report["windows"].items():
        h, p = part["history"], part["prompts"]
        print(
            f"{name:6s} updates {part['updates']:>7s}: reward {h['reward']:.3f}  KL(ref, k3) {h['kl_penalty']:.2e}  "
            f"grad norm {h['grad_norm']:.3f}  entropy {h['approx_entropy']:.3f}  lr {h['lr']:.1e}  "
            f"ratio [{h['probs_ratio_min']:.3f}, {h['probs_ratio_max']:.3f}]  "
            f"vLLM/trainer prob error {h['token_mult_prob_error']:.4f}  gen KL error {h['gen_kl_error']:.2e}  "
            f"answer tokens {h['mean_gen_tokens_per_sample']:.0f} (max {h['max_gen_tokens_per_sample']:.0f})  "
            f"truncated {h['truncation_rate']:.3f}  looped {h['looped']:.3f}  "
            f"zero-advantage prompts {p.get('zero_advantage_prompts', float('nan')):.2f}/16  "
            f"std<0.01 prompts {p.get('near_zero_spread_prompts', float('nan')):.2f}/16  "
            f"mean reward std {p.get('mean_reward_std', float('nan')):.4f}"
        )
    print(json.dumps(report["all_updates"]))
    print("validation:", json.dumps({s: {k: round(v, 4) for k, v in m.items()} for s, m in validation.items()}))


if __name__ == "__main__":
    main()
