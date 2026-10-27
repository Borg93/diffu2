"""The noise levels a MOSS-Image training step draws at: MOSS-Image's own recipe, the last interval of its technical
report's Stage 3 as KohakuMossTrainer documents and draws it (its default sampler, which the 1715 hand LoRA trained
with).

The levels are the 1,000-step training grid ``sigma_j = j / 1000`` (the grid the sampler steps on,
:func:`diffu_moss.modular_pipelines.moss.before_denoise.sampling_sigmas`); level ``j`` is drawn with the probability that
``S / a`` falls in ``((j - 1) / 1000, j / 1000]``, where ``S = 1 - sigmoid(Z)``, ``Z ~ N(-1.09, 0.95^2)``, truncated at
the cutoff ``a = 0.95``, mixed with 1% uniform over the grid; no shift. The median level is 0.78 and the quartiles 0.64
and 0.89: most steps are at the noisy end, where a page's layout and look are decided.
"""

from __future__ import annotations

import torch
from torch import Tensor

TRAIN_STEPS = 1000
MU = -1.09
SPREAD = 0.95
CUTOFF = 0.95
UNIFORM = 0.01


def sigma_grid(steps: int = TRAIN_STEPS) -> Tensor:
    """The training grid's noise levels ``j / steps``, ``j = 1 ... steps``, in fp64."""
    return torch.arange(1, steps + 1, dtype=torch.float64) / steps


def sigma_weights(steps: int = TRAIN_STEPS) -> Tensor:
    """The probability of each level of :func:`sigma_grid` (fp64, summing to 1)."""
    edges = CUTOFF * torch.arange(steps + 1, dtype=torch.float64) / steps
    logit = torch.log1p(-edges) - torch.log(edges)  # logit(1 - x): S <= x exactly when Z >= logit(1 - x)
    below = torch.special.ndtr((MU - logit) / SPREAD)
    truncated = (below[1:] - below[:-1]) / below[-1]
    return (1 - UNIFORM) * truncated + UNIFORM / steps


def draw_sigmas(count: int, generator: torch.Generator) -> Tensor:
    """``count`` noise levels drawn from :func:`sigma_weights` by ``generator`` (fp32, on the CPU)."""
    cumulative = sigma_weights().cumsum(0)
    picks = torch.searchsorted(cumulative, torch.rand(count, generator=generator, dtype=torch.float64))
    return sigma_grid()[picks.clamp(max=TRAIN_STEPS - 1)].float()
