"""Reference guidance: which passes it runs, on which caption, how it combines them, and that its reference-free pass
cannot outlive itself."""

import pytest
import torch

from diffu_page.guidance import ReferenceGuidance

CAPTIONS = {"encoder_hidden_states": (torch.tensor([1.0]), torch.tensor([0.0]))}  # the caption, the negative caption


class _ReferenceProbe(torch.nn.Module):
    """A denoiser that answers whether its call carried the reference tokens."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_cond_hidden_states: torch.Tensor | None = None,
        image_cond_ids: torch.Tensor | None = None,
    ) -> bool:
        return image_cond_hidden_states is not None


def _at_step(guider: ReferenceGuidance, step: int) -> ReferenceGuidance:
    guider.set_state(step=step, num_inference_steps=2, timestep=torch.tensor(1000.0 - 500.0 * step))
    return guider


def test_reference_guidance_pushes_from_the_unreferenced_pass_toward_the_reference_then_the_caption() -> None:
    guider = ReferenceGuidance(guidance_scale=2.0, reference_scale=3.0)

    output = guider.forward(
        pred_cond=torch.tensor([1.0]), pred_uncond=torch.tensor([0.5]), pred_unreferenced=torch.tensor([0.25])
    )

    assert output.pred.tolist() == [2.0]  # 0.25 + 3 * (0.5 - 0.25) + 2 * (1.0 - 0.5)


def test_reference_guidance_runs_its_reference_free_pass_on_the_negative_caption() -> None:
    guider = _at_step(ReferenceGuidance(guidance_scale=4.0, reference_scale=3.0), 0)
    passes = guider.prepare_inputs(CAPTIONS)
    for one_pass in passes:
        one_pass.noise_pred = one_pass.encoder_hidden_states  # each pass predicts the caption it was given

    output = guider(passes)

    assert output.pred.tolist() == [4.0]  # 0 + 3 * (0 - 0) + 4 * (1 - 0); on the caption it would be 2.0


def test_reference_guidance_at_reference_scale_one_runs_only_the_two_caption_passes() -> None:
    guider = _at_step(ReferenceGuidance(guidance_scale=4.0, reference_scale=1.0), 0)

    passes = guider.prepare_inputs(CAPTIONS)

    assert [one_pass.encoder_hidden_states.item() for one_pass in passes] == [1.0, 0.0]


def test_reference_guidance_restores_the_reference_after_a_pass_that_raised() -> None:
    guider, denoiser = _at_step(ReferenceGuidance(reference_scale=3.0), 0), _ReferenceProbe()
    for _ in range(3):  # the third, reference-free pass raises before the denoiser block calls cleanup_models
        guider.prepare_models(denoiser)
    _at_step(guider, 1)

    guider.prepare_models(denoiser)

    assert denoiser(torch.zeros(1), image_cond_hidden_states=torch.zeros(1), image_cond_ids=torch.zeros(1)) is True


def test_reference_guidance_refuses_its_third_pass_on_a_crop_without_a_reference() -> None:
    guider, denoiser = _at_step(ReferenceGuidance(reference_scale=3.0), 0), _ReferenceProbe()
    for _ in range(3):
        guider.prepare_models(denoiser)

    with pytest.raises(ValueError, match="without a reference"):
        denoiser(torch.zeros(1))
