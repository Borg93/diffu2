"""Guidance toward the reference image as well as the caption.

FLUX.3's classifier-free guidance compares the drawing with and without the caption. The reference image's tokens go
into both passes, so guidance never pushes toward the reference: the caption decides, and the reference barely
counts. :class:`ReferenceGuidance` adds a third pass, without the caption and without the reference, and combines the
three as InstructPix2Pix combines its text and image scales (Brooks et al., 2023)::

    pred = pred_unreferenced
           + reference_scale * (pred_uncond - pred_unreferenced)
           + guidance_scale * (pred_cond - pred_uncond)
"""

from __future__ import annotations

import math
from typing import override

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.guiders.guider_utils import GuiderOutput
from diffusers.modular_pipelines.modular_pipeline import BlockState
from torch.utils.hooks import RemovableHandle

REFERENCE_INPUTS = ("image_cond_hidden_states", "image_cond_ids")  # Flux3Transformer2DModel's reference tokens


class ReferenceGuidance(ClassifierFreeGuidance):
    """Classifier-free guidance on the caption and on the reference image.

    Diffusers' :class:`ClassifierFreeGuidance` with a third pass: the denoiser without the caption and without its
    reference inputs (:data:`REFERENCE_INPUTS`), removed by a forward pre-hook for that pass only, so FLUX.3-family
    denoiser blocks, which hand the reference to the transformer as those keyword arguments, work unchanged. At
    ``reference_scale`` 1 there is no third pass and this is classifier-free guidance exactly. ``guidance_rescale``
    rescales the caption's guidance only.

    Args:
        guidance_scale: How hard to push toward the caption.
        reference_scale: How hard to push toward the reference image.
        guidance_rescale: As :class:`ClassifierFreeGuidance`'s.
        use_original_formulation: As :class:`ClassifierFreeGuidance`'s.
        start: Fraction of the steps after which guidance starts.
        stop: Fraction of the steps after which guidance stops.
        enabled: Whether to guide at all; when not, only the caption's pass runs.

    Raises:
        ValueError: If the reference is guided while the caption is not (its pass without the caption never runs);
            from the denoiser's call, if the reference is guided on a crop without one (the third pass would cost a
            pass and change nothing).
    """

    _input_predictions = ["pred_cond", "pred_uncond", "pred_unreferenced"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 4.0,
        reference_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(guidance_scale, guidance_rescale, use_original_formulation, start, stop, enabled)
        unguided = 0.0 if use_original_formulation else 1.0  # the caption scale at which diffusers runs one pass
        if math.isclose(guidance_scale, unguided) and not math.isclose(reference_scale, 1.0):
            raise ValueError("reference guidance needs caption guidance: this guidance_scale runs no uncaptioned pass")
        self.reference_scale = reference_scale
        self._unreferenced_hook: RemovableHandle | None = None

    @override
    def prepare_inputs(self, data: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> list[BlockState]:
        batches = super().prepare_inputs(data)
        if not self._guides_reference():
            return batches
        return [*batches, self._prepare_batch(data, 1, "pred_unreferenced")]  # 1: the negative caption

    @override
    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        self.cleanup_models(denoiser)  # a pass that raised was never cleaned up; its hook must not outlive it
        super().prepare_models(denoiser)
        if self._guides_reference() and self._count_prepared == len(self._input_predictions):
            self._unreferenced_hook = denoiser.register_forward_pre_hook(_without_reference, with_kwargs=True)

    @override
    def cleanup_models(self, denoiser: torch.nn.Module) -> None:
        super().cleanup_models(denoiser)
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
        caption_guided = super().forward(pred_cond, pred_uncond)
        if pred_unreferenced is None:
            return caption_guided
        if pred_uncond is None:  # _guides_reference runs the reference-free pass only beside the uncaptioned one
            raise RuntimeError("a reference-free pass without an uncaptioned pass: reference guidance needs both")
        toward_reference = (self.reference_scale - 1.0) * (pred_uncond - pred_unreferenced)
        return GuiderOutput(pred=caption_guided.pred + toward_reference, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    @override
    def num_conditions(self) -> int:
        return super().num_conditions + int(self._guides_reference())

    def _guides_reference(self) -> bool:
        return self._is_cfg_enabled() and not math.isclose(self.reference_scale, 1.0)


def _without_reference(
    module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], dict[str, object]]:
    if all(kwargs.get(name) is None for name in REFERENCE_INPUTS):
        raise ValueError("reference guidance on a crop without a reference: draw it with reference_scale 1")
    return args, {name: value for name, value in kwargs.items() if name not in REFERENCE_INPUTS}
