"""The reward environment NeMo RL calls on every rollout: our page reward, each part logged beside it.

Follows ``nemotron/recipes/parse/dental/environment.py``. What reaches ``step`` is only the text of each answer
(``role`` and ``content``, decoded with special tokens skipped: ``nemo_rl/experience/rollouts.py``, ``calculate_rewards``
and ``generate_responses``) and the page's manifest row. Box and class tokens survive that decoding (they are ordinary
added tokens); ``</s>`` does not, and no finish reason is passed. So here the completion gate is loop detection only;
an answer cut at the token cap is zeroed by the trainer, which knows it (``grpo.reward_shaping`` in grpo_htr.yaml).
"""

from __future__ import annotations

import json
from pathlib import Path

import ray
import torch
from htr_reward import LOOP, WEIGHTS, components, loop_rule
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

LOGGED = ("reward", "text", "cer", "wer", "boxes", "box_f1", "predicted_boxes", "looped", "gate")


@ray.remote(num_cpus=0.25)
class HtrEnvironment(EnvironmentInterface):
    """Scores answers against the page's target; ``cfg``: ``metrics_file``, ``phase``, optional ``weights``, ``loop``."""

    def __init__(self, cfg):
        self.path = Path(cfg["metrics_file"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.phase = cfg["phase"]
        self.weights = {name: float(weight) for name, weight in (cfg.get("weights") or WEIGHTS).items()}
        self.loop = loop_rule(cfg.get("loop") or LOOP)

    def shutdown(self):
        pass

    def step(self, message_log_batch, metadata, return_extracted_answer=False):
        texts, values = [], []
        for log, row in zip(message_log_batch, metadata, strict=True):
            text = "".join(message["content"] for message in log if message["role"] == "assistant")
            texts.append(text)
            values.append(components(text, row["target"], weights=self.weights, loop=self.loop))
        with self.path.open("a") as stream:
            samples = [{key: value[key] for key in LOGGED} for value in values]
            stream.write(json.dumps({"phase": self.phase, "samples": samples}) + "\n")
        rewards = torch.tensor([value["reward"] for value in values], dtype=torch.float32)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": f"Reward: {r:g}"} for r in rewards],
            metadata=metadata,
            next_stop_strings=[None] * len(rewards),
            rewards=rewards,
            terminateds=torch.ones_like(rewards),
            answers=texts if return_extracted_answer else None,
        )

    def global_post_process_and_metrics(self, batch):
        return batch, {"reward": batch["rewards"].mean().item()}
