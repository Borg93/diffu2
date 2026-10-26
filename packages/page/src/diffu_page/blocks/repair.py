"""Line repair inside the denoising loop: a drawn crop's rejected lines drawn again in place.

Given a repair mask, the crop is drawn again from new noise, and after every scheduler step the latents outside the mask
are set back to the drawn crop's own latents at that step's noise level (flow matching: ``(1 - sigma) * drawn + sigma *
noise``, the scheduler's ``scale_noise``, from the one initial noise as diffusers' FLUX inpainting does). The masked
lines are redrawn in the context of the page around them, and everything else ends where it was. Without a mask both
steps do nothing. :func:`diffu_page.pipeline.crop_blocks` puts them into the model's own denoise blocks, the blend into
its denoise loop after the scheduler step; :func:`diffu_page.pipeline.repair_lines` decides which lines to redraw and
which redrawn lines to keep.
"""

from __future__ import annotations

from typing import override

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.modular_pipelines import (
    ComponentSpec,
    InputParam,
    ModularPipeline,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline import BlockState


def _tokens(grid: torch.Tensor) -> torch.Tensor:
    """``(B, C, h, w)`` on the latent grid as tokens ``(B, h * w, C)``, in the order FLUX.3 packs its latents."""
    return grid.flatten(2).transpose(1, 2)


class RepairPrepareStep(ModularPipelineBlocks):
    """The drawn crop's latents and the repair mask as tokens beside the initial noise, which the blend keeps; in
    float32, as the scheduler steps in it."""

    @property
    @override
    def description(self) -> str:
        return (
            "Prepares a line repair: the drawn crop's latents and the repair mask as tokens, and a copy of the initial "
            "noise. Does nothing without a repair mask."
        )

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor, description="The initial noise as tokens."),
            InputParam(
                "repair_latents",
                type_hint=torch.Tensor,
                description="The drawn crop's latents (1, C, h, w), as the model's own VAE encoder makes them.",
            ),
            InputParam(
                "repair_mask",
                type_hint=torch.Tensor,
                description="Where to draw again (1) and where to keep the drawn crop (0), on the latent grid (1, 1, h, w).",
            ),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("repair_drawn", type_hint=torch.Tensor, description="The drawn crop's latents as tokens."),
            OutputParam(
                "repair_noise", type_hint=torch.Tensor, description="The initial noise the redraw starts from."
            ),
            OutputParam("repair_weight", type_hint=torch.Tensor, description="The repair mask as tokens (B, seq, 1)."),
        ]

    @torch.no_grad()
    def __call__(self, components: ModularPipeline, state: PipelineState) -> tuple[ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.repair_drawn = block_state.repair_noise = block_state.repair_weight = None
        if block_state.repair_mask is not None:
            latents: torch.Tensor = block_state.latents
            if block_state.repair_latents is None:
                raise ValueError("a repair mask needs the drawn crop's latents (repair_latents)")
            drawn = _tokens(block_state.repair_latents).to(latents.device, torch.float32)
            weight = _tokens(block_state.repair_mask).to(latents.device, torch.float32)
            if drawn.shape != latents.shape or weight.shape != (*latents.shape[:2], 1):
                raise ValueError(
                    f"the drawn crop's latents {tuple(drawn.shape)} and repair mask {tuple(weight.shape)} do not fit "
                    f"the latents being drawn {tuple(latents.shape)}"
                )
            block_state.repair_drawn, block_state.repair_noise, block_state.repair_weight = (
                drawn,
                latents.to(torch.float32, copy=True),
                weight,
            )
        self.set_block_state(state, block_state)
        return components, state


class RepairBlendStep(ModularPipelineBlocks):
    """After a scheduler step, the latents outside the repair mask set back to the drawn crop's at the new noise level."""

    @property
    @override
    def expected_components(self) -> list[ComponentSpec]:
        # config-less, as in diffusers' own loop steps: the scheduler is owned by the timesteps step
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler, default_creation_method="from_config")]

    @property
    @override
    def description(self) -> str:
        return (
            "Step within the denoising loop, after the scheduler step: keeps the latents outside the repair mask on the "
            "drawn crop's latents, noised to the step's level. Does nothing without a repair mask."
        )

    @property
    @override
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("repair_drawn", type_hint=torch.Tensor, description="The drawn crop's latents as tokens."),
            InputParam("repair_noise", type_hint=torch.Tensor, description="The initial noise the redraw starts from."),
            InputParam("repair_weight", type_hint=torch.Tensor, description="The repair mask as tokens (B, seq, 1)."),
        ]

    @property
    @override
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="The denoised latents")]

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
    ) -> tuple[ModularPipeline, BlockState]:
        if block_state.repair_weight is not None:
            scheduler: FlowMatchEulerDiscreteScheduler = components.scheduler
            latents: torch.Tensor = block_state.latents
            # the level this step reached: set_timesteps ends the sigmas on 0, so the last step keeps the drawn crop
            sigma = scheduler.sigmas[scheduler.step_index].to(latents.device, torch.float32)
            kept = (1 - sigma) * block_state.repair_drawn + sigma * block_state.repair_noise
            block_state.latents = torch.lerp(kept, latents.float(), block_state.repair_weight).to(latents.dtype)
        return components, block_state
