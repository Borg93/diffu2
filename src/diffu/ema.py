"""EMA weights: diffusers' ``EMAModel`` with a one-pass foreach update."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from diffusers.training_utils import EMAModel


def _deepspeed_zero3() -> bool:
    """ZeRO-3 shards parameters, which the parent's step gathers first — leave that case to it."""
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    return is_deepspeed_zero3_enabled()


class LerpEMAModel(EMAModel):
    """``EMAModel`` whose foreach step is one in-place ``lerp`` per parameter list.

    diffusers' foreach step computes ``shadow -= (1 - decay) * (shadow - param)`` as ``_foreach_sub``
    (a full temporary copy of the shadow weights) then ``_foreach_sub_``. ``_foreach_lerp_`` is the same update
    in one pass: the step is bit-identical whenever ``1 - decay < 0.5`` (lerp's formula is then exactly
    ``shadow + w * (param - shadow)``), i.e. every step after the first warmup steps, whose rounding-level
    differences are all that ever separates the two shadows.
    Everything else — decay schedule, state dict, store/restore/copy_to — is the parent's.
    """

    @torch.no_grad()  # as the parent's: the shadow update must not be recorded by autograd
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        if not self.foreach or isinstance(parameters, torch.nn.Module) or _deepspeed_zero3():
            return super().step(parameters)
        params = list(parameters)
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        trained = [p for p in params if p.requires_grad]
        shadows = [s for s, p in zip(self.shadow_params, params, strict=True) if p.requires_grad]
        frozen = [p for p in params if not p.requires_grad]
        if frozen:
            frozen_shadows = [
                s for s, p in zip(self.shadow_params, params, strict=True) if not p.requires_grad
            ]
            torch._foreach_copy_(frozen_shadows, frozen, non_blocking=True)
        if trained:
            torch._foreach_lerp_(shadows, trained, 1.0 - decay)
