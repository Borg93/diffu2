"""Rectified flow / flow matching objective (SD3 formulation), native PyTorch.

Convention: predict velocity v = eps - x0 along the linear interpolant
    x_t = (1 - t) * x0 + t * eps ,   t in [0, 1]  (t=0 data, t=1 noise).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_density_for_timestep_sampling


def repa_loss(proj_hidden: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
    """REPA (ICLR'25): cosine-align projected DiT hidden states to frozen-encoder features.

    Args:
        proj_hidden: Projected DiT hidden states, ``[B, D]`` or ``[B, N, D]``.
        target_feats: Frozen-encoder (DINOv3) features, same shape as ``proj_hidden``.

    Returns:
        Scalar alignment loss (lower is better fine structure).
    """
    p = F.normalize(proj_hidden, dim=-1)
    t = F.normalize(target_feats, dim=-1)
    return (1.0 - (p * t).sum(dim=-1)).mean()


def sample_timesteps(batch: int, device: torch.device, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Logit-normal flow-matching timesteps ``[B]`` in ``[0, 1]`` (emphasizes the hard mid-range).

    Delegates to diffusers' own SD3 sampler instead of re-deriving the formula: with
    ``weighting_scheme="logit_normal"`` it returns ``sigmoid(N(mean, std))`` — identical to the
    hand-rolled ``sigmoid(randn*std + mean)`` it replaces.
    """
    return compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal", batch_size=batch, logit_mean=mean, logit_std=std, device=device
    )


def interpolate(x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear interpolant ``x_t`` and its velocity target ``v = eps - x0``.

    Args:
        x0: Clean data latents ``[B, ...]`` (t=0 endpoint).
        eps: Gaussian noise of the same shape (t=1 endpoint).
        t: Per-sample timesteps ``[B]`` in ``[0, 1]``.

    Returns:
        ``(x_t, target_v)``, both shaped like ``x0``.
    """
    t_ = t.view(-1, *([1] * (x0.dim() - 1)))
    x_t = (1 - t_) * x0 + t_ * eps
    target_v = eps - x0
    return x_t, target_v


def flow_loss(
    v_pred: torch.Tensor, target_v: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """MSE on velocity. ``weight`` (broadcastable, e.g. ink-focal map) is applied as a weighted mean,
    normalized by the mean weight so the loss magnitude stays comparable to the unweighted case."""
    se = (v_pred - target_v) ** 2
    if weight is not None:
        return (se * weight).mean() / weight.mean().clamp_min(1e-6)
    return se.mean()
