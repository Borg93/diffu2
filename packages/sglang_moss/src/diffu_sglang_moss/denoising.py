"""SGLang's MOSS-Image denoising stage with ``diffu_moss``'s guidance stop.

Our pipeline's guider (diffusers' ``ClassifierFreeGuidance`` with ``stop``, the page pipeline's ``guidance_stop``)
guides the steps before ``int(stop * steps)`` and none after: from that step on only the conditional pass runs, and
its prediction is the step's velocity as it is. SGLang's own ``cfg_gate_step`` is another thing: it keeps guiding
with the last conditional-minus-unconditional difference. A request's ``guidance_stop`` (1, every step guided, by
default) asks for ours.
"""

from __future__ import annotations

import math
from dataclasses import replace  # noqa: TID251  SGLang's CFG policy is a dataclass
from typing import Any

import torch
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.moss_image import (
    MOSSImageDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from torch import Tensor


def guidance_stopped(guidance_stop: float, step: int, steps: int) -> bool:
    """Whether step ``step`` of ``steps`` is past ``guidance_stop``, as diffusers' guider counts it."""
    return step >= int(guidance_stop * steps)


class DiffuMOSSImageDenoisingStage(MOSSImageDenoisingStage):
    """SGLang's MOSS-Image denoising stage whose steps past the request's ``guidance_stop`` run the conditional pass
    alone and take its prediction as the velocity.

    Raises:
        ValueError: From a step, if ``guidance_stop`` is not in [0, 1], or is below 1 under CFG parallelism (whose
            ranks each run one pass).
    """

    def _predict_noise_with_cfg(
        self,
        current_model: Any,
        latent_model_input: Tensor,
        timestep: Any,
        batch: Req,
        timestep_index: int,
        attn_metadata: Any,
        target_dtype: torch.dtype,
        current_guidance_scale: Any,
        cfg_policy: CFGPolicy,
        cfg_gate_state: dict[str, Any] | None,
        server_args: ServerArgs,
        guidance: Tensor,
        latents: Tensor,
    ) -> Tensor | tuple[Tensor, ...]:
        stop = float(getattr(batch.sampling_params, "guidance_stop", 1.0))
        if not (math.isfinite(stop) and 0.0 <= stop <= 1.0):
            raise ValueError(f"guidance_stop must be in [0, 1], not {stop}")
        if stop < 1.0 and batch.do_classifier_free_guidance:
            if server_args.enable_cfg_parallel:
                raise ValueError("guidance_stop below 1 is not supported under CFG parallelism")
            if guidance_stopped(stop, timestep_index, len(batch.timesteps)):
                conditional = [branch for branch in cfg_policy.branches if branch.is_conditional]
                cfg_policy, cfg_gate_state = replace(cfg_policy, branches=conditional), None
        return super()._predict_noise_with_cfg(
            current_model=current_model,
            latent_model_input=latent_model_input,
            timestep=timestep,
            batch=batch,
            timestep_index=timestep_index,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            current_guidance_scale=current_guidance_scale,
            cfg_policy=cfg_policy,
            cfg_gate_state=cfg_gate_state,
            server_args=server_args,
            guidance=guidance,
            latents=latents,
        )
