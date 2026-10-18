"""Recognizer-reward fine-tuning (Flow-OPD style) — POST-BASE lever. See docs/RL_PLAN.md.

Build/run only after a base run (e.g. exp_sd35_inkfocal) lands a legible model to fine-tune. The leaf
utilities below are implemented + unit-tested; the model-coupled pieces (``sde_sampler``, ``sft``) are
scaffolded and raise ``NotImplementedError`` until that base exists.
"""

from .advantage import PerPromptAdvantage
from .reward import htr_cer_reward
from .sde_logprob import sde_step_with_logprob

__all__ = ["PerPromptAdvantage", "htr_cer_reward", "sde_step_with_logprob"]
