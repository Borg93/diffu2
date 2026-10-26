"""A glyph model's NeMo adapter, in NeMo AutoModel's environment (skipped elsewhere, where NeMo is not installed): each
token's glyph map beside its noisy latents, a loss that starts as the trained model's own text-to-image loss and trains
the glyph map's columns, refusals of a batch without a fitting map, and the adapter a repo is trained with."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("nemo_automodel")

from diffusers import Flux3Transformer2DModel
from nemo_automodel.components.flow_matching.adapters import FlowMatchingContext, Flux3Adapter
from nemo_automodel.components.flow_matching.pipeline import FlowMatchingPipeline

from diffu.configs import DiffuConfig
from diffu.repo import ModelIndex
from diffu_train.nemo import GlyphConcatAdapter, adapter_for

CPU = torch.device("cpu")


def _context(noisy: torch.Tensor, glyph_maps: list[torch.Tensor] | None) -> FlowMatchingContext:
    """One pair's training step as NeMo's flow matching hands it to its adapter: its noisy latents at sigma 0.5, a
    caption of three tokens, and its context latents (the glyph map) if it has any."""
    batch: dict[str, object] = {"image_latents": noisy, "text_embeddings": torch.randn(1, 3, 16), "data_type": "image"}
    if glyph_maps is not None:
        batch["context_latents"] = glyph_maps
    return FlowMatchingContext(
        noisy_latents=noisy,
        latents=noisy,
        timesteps=torch.tensor([500.0]),
        sigma=torch.tensor([0.5]),
        task_type="t2v",
        data_type="image",
        device=CPU,
        dtype=torch.float32,
        batch=batch,
    )


def test_glyph_adapter_puts_each_tokens_cell_of_the_glyph_map_beside_its_noisy_latents() -> None:
    noisy, glyph_map = torch.randn(1, 16, 2, 3), torch.randn(1, 16, 2, 3)

    inputs = GlyphConcatAdapter().prepare_inputs(_context(noisy, [glyph_map]))

    tokens = [latents.permute(0, 2, 3, 1).reshape(1, 6, 16) for latents in (noisy, glyph_map)]  # row by row
    assert torch.equal(inputs["hidden_states"], torch.cat(tokens, dim=-1))
    assert not {"image_cond_hidden_states", "image_cond_ids"} & set(inputs)


def _flow_matching(adapter: Flux3Adapter) -> FlowMatchingPipeline:
    """NeMo's flow matching as E2b and E2c train: logit-normal timesteps at shift 3, no loss weighting, no text
    dropout of its own."""
    return FlowMatchingPipeline(
        model_adapter=adapter,
        timestep_sampling="logit_normal",
        flow_shift=3.0,
        mix_uniform_ratio=0.0,
        cfg_dropout_prob=0.0,
        use_loss_weighting=False,
        device=CPU,
    )


def _widened(trained: Flux3Transformer2DModel, config: DiffuConfig) -> Flux3Transformer2DModel:
    """``trained`` as a glyph model of ``config``'s: the glyph map's input columns and output rows zero."""
    glyph_model = Flux3Transformer2DModel.from_config(config.model_copy(update={"glyph": True}).transformer_config())
    weights = trained.state_dict()
    image_input, output = weights["x_embedders.image.weight"], weights["proj_out.weight"]
    glyph_model.load_state_dict(
        {
            **weights,
            "x_embedders.image.weight": torch.cat([image_input, torch.zeros_like(image_input)], dim=1),
            "proj_out.weight": torch.cat([output, torch.zeros_like(output)], dim=0),
        }
    )
    return glyph_model


def test_glyph_adapter_starts_a_widened_model_at_its_trained_models_loss_and_trains_the_glyph_maps_columns(
    tiny_diffu: Path, tiny_diffu_config: DiffuConfig
) -> None:
    trained = Flux3Transformer2DModel.from_pretrained(tiny_diffu, subfolder="transformer")
    widened = _widened(trained, tiny_diffu_config)
    torch.manual_seed(0)
    pair = {"image_latents": torch.randn(1, 16, 2, 3), "text_embeddings": torch.randn(1, 3, 16), "data_type": "image"}
    glyph_map = torch.randn(1, 16, 2, 3)
    torch.manual_seed(1)  # the same timestep and noise for both
    text_to_image = _flow_matching(Flux3Adapter()).step(
        trained, pair, device=CPU, dtype=torch.float32, collect_metrics=False, check_loss=False
    )[1]
    torch.manual_seed(1)

    loss = _flow_matching(GlyphConcatAdapter()).step(
        widened,
        {**pair, "context_latents": [glyph_map]},
        device=CPU,
        dtype=torch.float32,
        collect_metrics=False,
        check_loss=False,
    )[1]
    loss.backward()

    assert torch.equal(loss, text_to_image)
    assert torch.count_nonzero(widened.x_embedders["image"].weight.grad[:, 16:]) > 0  # the glyph map's columns learn
    assert torch.count_nonzero(widened.proj_out.weight.grad[16:]) == 0  # the rows the loss never reads do not


@pytest.mark.parametrize(
    ("glyph_maps", "refusal"),
    [
        pytest.param(None, "context latents is the glyph map: this has none", id="no-glyph-map"),
        pytest.param([torch.zeros(1, 16, 3, 2)], r"on a \(3, 2\) grid, the target's on \(2, 3\)", id="another-grid"),
    ],
)
def test_glyph_adapter_refuses_a_batch_without_a_glyph_map_on_its_targets_grid(
    glyph_maps: list[torch.Tensor] | None, refusal: str
) -> None:
    context = _context(torch.randn(1, 16, 2, 3), glyph_maps)

    with pytest.raises(ValueError, match=refusal):
        GlyphConcatAdapter().prepare_inputs(context)


@pytest.mark.parametrize(
    ("repo", "adapter"),
    [
        pytest.param("tiny_diffu_glyph", GlyphConcatAdapter, id="glyph-model"),
        pytest.param("tiny_diffu", type(None), id="model-without-the-glyph-map"),
    ],
)
def test_adapter_for_picks_the_glyph_adapter_for_a_glyph_model_only(
    repo: str, adapter: type, request: pytest.FixtureRequest
) -> None:
    index = ModelIndex.read(request.getfixturevalue(repo))

    assert type(adapter_for(index, "flux3")) is adapter


def test_adapter_for_refuses_a_glyph_model_another_adapter_than_flux3s(tiny_diffu_glyph: Path) -> None:
    index = ModelIndex.read(tiny_diffu_glyph)

    with pytest.raises(ValueError, match="adapter_type flux3, not 'qwen_image'"):
        adapter_for(index, "qwen_image")
