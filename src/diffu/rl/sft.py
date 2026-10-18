"""Advantage-weighted flow-matching SFT loop (Flow-OPD "variant A") — recognizer-reward fine-tuning.

SCAFFOLD (not implemented): the loop structure is documented in :func:`run_sft`; the body raises
``NotImplementedError`` until a base model exists to fine-tune. Mirrors Flow-OPD's ``train_sd3_sft.py`` —
NOT importance-sampling PPO; the SDE log-probs are collected for diversity but unused in this loss.
See docs/RL_PLAN.md.
"""

from __future__ import annotations


def run_sft(
    base_ckpt: str,
    data_dir: str,
    out_dir: str,
    *,
    k: int = 16,
    beta: float = 0.04,
    lr: float = 1e-4,
    steps: int = 10_000,
    reward_recognizer: str = "",
    noise_level: float = 0.7,
) -> None:
    """Fine-tune a *legible base* with advantage-weighted flow-matching SFT.

    Per outer epoch (see Flow-OPD ``train_sd3_sft.py``):

    1. ``sde_sampler.sample_group`` → ``k`` diverse rollouts per ``(text, style)``.
    2. ``reward.htr_cer_reward`` with a **held-out** recognizer (``reward_recognizer`` ≠ the eval gauge).
    3. ``advantage.PerPromptAdvantage(mode='sft')`` → best-of-group advantage.
    4. Re-noise each sampled latent; ``loss = (advantage * flow_loss + beta * kl_to_base).mean()`` where
       ``kl_to_base = MSE(learner_velocity, base_velocity)`` (LoRA ``disable_adapter`` gives the base for
       free; a full fine-tune needs a frozen base copy).
    5. ``opt.step()`` + EMA update.

    Gate: does Swedish ``gen_cer_cfg`` beat ``base_ckpt``? If not, RL isn't worth the compute.
    """
    raise NotImplementedError(
        "RL SFT loop is post-base scaffolding. Build after a base run (e.g. exp_sd35_inkfocal) lands a "
        "legible model; reuse src/diffu/rl/{sde_sampler,reward,advantage}.py. See docs/RL_PLAN.md. "
        f"(intended config: base_ckpt={base_ckpt!r}, k={k}, beta={beta}, lr={lr}, steps={steps}, "
        f"reward_recognizer={reward_recognizer!r}, noise_level={noise_level}, data_dir={data_dir!r}, "
        f"out_dir={out_dir!r})"
    )
