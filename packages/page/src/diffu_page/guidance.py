"""Guidance toward the reference image as well as the caption.

FLUX.3's classifier-free guidance compares the drawing with and without the caption. The reference image's tokens go
into both passes, so guidance never pushes toward the reference: the caption decides, and the reference barely counts
(the control probe of 23 Oct 2026). :class:`ReferenceGuidance` adds a third pass, without the caption and without the
reference, and combines the three as InstructPix2Pix combines its text and image scales (Brooks et al., 2023)::

    pred = pred_unreferenced
           + reference_scale * (pred_uncond - pred_unreferenced)
           + guidance_scale * (pred_cond - pred_uncond)
"""

from __future__ import annotations

import math
from typing import override

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.guiders.guider_utils import BaseGuidance, GuiderOutput
from diffusers.modular_pipelines.modular_pipeline import BlockState
from torch.utils.hooks import RemovableHandle

REFERENCE_INPUTS = ("image_cond_hidden_states", "image_cond_ids")  # Flux3Transformer2DModel's reference tokens


class ReferenceGuidance(BaseGuidance):
    """Classifier-free guidance on the caption and on the reference image.

    The third pass runs the denoiser without its reference inputs (:data:`REFERENCE_INPUTS`), removed by a forward
    pre-hook for that pass only, so any denoiser block that hands the guider's passes to the transformer works
    unchanged. Each scale at 1 leaves its pass out: with ``reference_scale`` 1 this is plain classifier-free guidance.

    Args:
        guidance_scale: How hard to push toward the caption.
        reference_scale: How hard to push toward the reference image.
        start: Fraction of the steps after which guidance starts.
        stop: Fraction of the steps after which guidance stops.
        enabled: Whether to guide at all; when not, only the caption's pass runs.
    """

    _input_predictions = ["pred_cond", "pred_uncond", "pred_unreferenced"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 4.0,
        reference_scale: float = 1.0,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(start, stop, enabled)
        self.guidance_scale = guidance_scale
        self.reference_scale = reference_scale
        self._unreferenced_hook: RemovableHandle | None = None

    @override
    def prepare_inputs(self, data: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> list[BlockState]:
        caption_index = {"pred_cond": 0, "pred_uncond": 1, "pred_unreferenced": 1}  # 1: the negative (empty) caption
        return [self._prepare_batch(data, caption_index[name], name) for name in self._passes()]

    @override
    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        super().prepare_models(denoiser)
        if self._passes()[self._count_prepared - 1] == "pred_unreferenced":
            self._unreferenced_hook = denoiser.register_forward_pre_hook(_without_reference, with_kwargs=True)

    @override
    def cleanup_models(self, denoiser: torch.nn.Module) -> None:
        if self._unreferenced_hook is not None:
            self._unreferenced_hook.remove()
            self._unreferenced_hook = None

    @override
    def forward(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor | None = None,
        pred_unreferenced: torch.Tensor | None = None,
    ) -> GuiderOutput:
        if pred_uncond is None:
            return GuiderOutput(pred=pred_cond, pred_cond=pred_cond, pred_uncond=None)
        toward_caption = self.guidance_scale * (pred_cond - pred_uncond)
        if pred_unreferenced is None:
            return GuiderOutput(pred=pred_uncond + toward_caption, pred_cond=pred_cond, pred_uncond=pred_uncond)
        toward_reference = self.reference_scale * (pred_uncond - pred_unreferenced)
        pred = pred_unreferenced + toward_reference + toward_caption
        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    @override
    def is_conditional(self) -> bool:
        return self._count_prepared == 1

    @property
    @override
    def num_conditions(self) -> int:
        return len(self._passes())

    def _passes(self) -> list[str]:
        """The predictions this step needs, in the order the denoiser runs them."""
        if not self._guiding_this_step():
            return self._input_predictions[:1]
        if not math.isclose(self.reference_scale, 1.0):
            return self._input_predictions
        if not math.isclose(self.guidance_scale, 1.0):
            return self._input_predictions[:2]
        return self._input_predictions[:1]

    def _guiding_this_step(self) -> bool:
        if not self._enabled:
            return False
        if self._num_inference_steps is None:
            return True
        first, stop = int(self._start * self._num_inference_steps), int(self._stop * self._num_inference_steps)
        return first <= self._step < stop


def _without_reference(
    module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], dict[str, object]]:
    return args, {name: value for name, value in kwargs.items() if name not in REFERENCE_INPUTS}
