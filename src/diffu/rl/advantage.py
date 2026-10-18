"""Group-relative advantage for recognizer-reward fine-tuning (adapted from Flow-OPD stat_tracking.py).

Samples are grouped by prompt (the same ``(text, style)`` rendered K times via the SDE sampler). The
advantage tells the update which samples in a group to push toward / away from — no value network (the "G"
in GRPO). See docs/RL_PLAN.md.
"""

from __future__ import annotations

import numpy as np


class PerPromptAdvantage:
    """Turn per-sample rewards into per-sample advantages, grouped by prompt.

    ``mode='sft'``  → **best-of-group**: 1.0 for the max-reward sample(s) in each prompt's group, else 0.0
        (OnlineSFT / hard reward-weighted regression — what the pasted ``train_sd3_sft.py`` uses).
    ``mode='grpo'`` → **normalized**: ``(reward − group_mean) / (std + eps)`` (needs ≥2 samples/group to be
        meaningful). ``global_std`` divides by the std over the WHOLE batch (lower variance) vs per-group.
    """

    def __init__(self, mode: str = "sft", *, global_std: bool = True, eps: float = 1e-4) -> None:
        if mode not in ("sft", "grpo"):
            raise ValueError(f"mode must be 'sft' or 'grpo', got {mode!r}")
        self.mode = mode
        self.global_std = global_std
        self.eps = eps

    def __call__(self, prompts: list[str], rewards: list[float]) -> np.ndarray:
        """``prompts``/``rewards``: parallel lists of length N. Returns advantages ``[N]`` in input order."""
        if len(prompts) != len(rewards):
            raise ValueError(f"prompts ({len(prompts)}) and rewards ({len(rewards)}) length mismatch")
        prompt_arr = np.asarray(prompts)
        reward_arr = np.asarray(rewards, dtype=np.float64)
        adv = np.zeros_like(reward_arr)  # [N]
        for prompt in np.unique(prompt_arr):
            in_group = prompt_arr == prompt
            group = reward_arr[in_group]
            if self.mode == "sft":
                adv[in_group] = (group == group.max()).astype(np.float64)  # best-of-group indicator
            else:  # grpo: normalize by group mean, std over batch (global) or group
                std = (reward_arr.std() if self.global_std else group.std()) + self.eps
                adv[in_group] = (group - group.mean()) / std
        return adv
