"""Reference guidance: how its three passes combine."""

import torch

from diffu_page.guidance import ReferenceGuidance


def test_reference_guidance_pushes_from_the_unreferenced_pass_toward_the_reference_then_the_caption() -> None:
    guider = ReferenceGuidance(guidance_scale=2.0, reference_scale=3.0)

    output = guider.forward(
        pred_cond=torch.tensor([1.0]), pred_uncond=torch.tensor([0.5]), pred_unreferenced=torch.tensor([0.25])
    )

    assert output.pred.tolist() == [2.0]  # 0.25 + 3 * (0.5 - 0.25) + 2 * (1.0 - 0.5)
