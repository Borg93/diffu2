"""SDE denoise step with a tractable log-probability (adapted from Flow-OPD sd3_sde_with_logprob.py).

A flow-matching ODE step is deterministic → no probability → no policy gradient. Converting the step into
an SDE (inject per-step Gaussian noise) makes each step a Gaussian, so its log-density is tractable. Two
uses: (1) the noise gives DIVERSE samples per prompt (needed for group advantage — used by SFT *and* GRPO);
(2) the returned ``log_prob`` is only needed for the importance-sampling GRPO variant (not the SFT loss).
Scheduler-only: it reads ``scheduler.sigmas`` + the model's predicted velocity. See docs/RL_PLAN.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from diffusers.utils.torch_utils import randn_tensor

if TYPE_CHECKING:
    from diffusers import FlowMatchEulerDiscreteScheduler


def sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    velocity: torch.Tensor,  # [B, C, H, W] model-predicted flow velocity
    timestep: torch.Tensor,  # [B] current timesteps
    sample: torch.Tensor,  # [B, C, H, W] current latents x_t
    *,
    noise_level: float = 0.7,
    prev_sample: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One reverse SDE step. Returns ``(prev_sample, log_prob, prev_sample_mean, std_dev_t)``.

    ``prev_sample`` given → score that transition (log_prob of an already-taken step, for the importance
    ratio); ``None`` → sample a new ``prev_sample``. ``log_prob`` is ``[B]`` (mean over non-batch dims). All
    math runs in fp32 — bf16 overflows when forming ``prev_sample_mean``.
    """
    velocity = velocity.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [s + 1 for s in step_index]
    shape = (-1, *([1] * (sample.ndim - 1)))
    sigma = scheduler.sigmas[step_index].view(shape)
    sigma_prev = scheduler.sigmas[prev_step_index].view(shape)
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma  # negative (sigma decreases toward 0)

    # sigma==1 (pure noise) would divide by zero -> substitute sigma_max there.
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + velocity * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    std = std_dev_t * torch.sqrt(-dt)  # per-step Gaussian std
    if prev_sample is None:
        noise = randn_tensor(
            velocity.shape, generator=generator, device=velocity.device, dtype=velocity.dtype
        )
        prev_sample = prev_sample_mean + std * noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std**2)
        - torch.log(std)
        - math.log(math.sqrt(2 * math.pi))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))  # [B]
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
