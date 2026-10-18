"""SDE rollout sampler — generate K diverse samples per ``(text, style)`` for group-relative RL.

SCAFFOLD (not implemented): this must drive ``Diffu``'s denoise loop while injecting
``sde_step_with_logprob`` at each step (and exposing the predicted velocity per step), so the K samples for
one prompt actually differ — that spread is what the advantage ranks. Build once a base model exists.
See docs/RL_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image as PILImage

    from ..config import Config
    from ..model import Diffu


@dataclass
class Rollout:
    """One SDE rollout: the final image + the trajectory pieces RL consumes."""

    image: PILImage
    latents: list[torch.Tensor]  # x_T … x_0, one per denoise step
    log_probs: list[torch.Tensor]  # per-step log π — only the GRPO variant uses these (SFT ignores them)


def sample_group(
    model: Diffu,
    cfg: Config,
    text: str,
    style: torch.Tensor,  # [1, 3, S, S]
    *,
    k: int,
    num_steps: int = 24,
    noise_level: float = 0.7,
    cfg_scale: float = 0.0,
) -> list[Rollout]:
    """Generate ``k`` DIVERSE rollouts for one ``(text, style)`` via SDE sampling.

    SCAFFOLD — wire ``sde_step_with_logprob`` into ``Diffu.denoise`` (replace the deterministic scheduler
    step with the SDE step at ``noise_level``) once ``exp_sd35_inkfocal`` yields a legible base.
    """
    raise NotImplementedError(
        "RL SDE sampler is post-base scaffolding: inject sde_step_with_logprob into Diffu.denoise once a "
        "legible base exists. See docs/RL_PLAN.md."
    )
