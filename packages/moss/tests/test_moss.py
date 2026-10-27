"""MOSS-Image on CPU: our DiT predicts bit for bit the velocity the verified implementation predicted when it is given
no reference, and with a reference reads its tokens at timestep 0 on their own grid and reference axis; the text
encoder refuses a prompt past its 4,096 tokens, and a MOSS repo draws with the DiT folder it names."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import pytest
import torch
from diffusers.modular_pipelines import ModularPipeline, SequentialPipelineBlocks
from torch import nn

from diffu.repo import ModelIndex
from diffu_moss.init import write_moss_repo
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.modular_pipelines.moss.encoders import MossTextEncoderStep
from diffu_moss.transformer import MossTransformer, token_positions

# The verified implementation's velocities (the scratchpad's moss_model.MossTransformer, from which this DiT was
# ported, 27 Oct) for the weights and inputs below, with every block and with block 1 skipped; float32 values as JSON
VELOCITIES = Path(__file__).parent / "data" / "transformer_velocity.json"
TINY_DIT = {
    "dim": 32,
    "ffn_dim": 16,
    "num_heads": 2,
    "num_layers": 3,
    "text_dim": 8,
    "freq_dim": 8,
    "latent_channels": 4,
}


def _seeded_dit() -> MossTransformer:
    """The tiny DiT, each weight drawn from a generator seeded by the CRC-32 of its name (so in any module order)."""
    model = MossTransformer(**TINY_DIT).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            generator = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
    return model


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two samples' latents on a grid of 3 rows and 5 columns, prompts of 6 and 3 tokens and their mask."""
    latents = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(1))
    text = torch.randn(2, 6, 8, generator=torch.Generator().manual_seed(2))
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    return latents, text, mask


@pytest.mark.parametrize(
    ("skip_layers", "expected"),
    [
        pytest.param(frozenset(), "velocity", id="every-block"),
        pytest.param(frozenset({1}), "velocity_skipping_1", id="block-1-skipped"),
    ],
)
def test_moss_transformer_without_references_predicts_exactly_the_velocity_the_verified_implementation_predicted(
    skip_layers: frozenset[int], expected: str
) -> None:
    model = _seeded_dit()
    latents, text, mask = _inputs()

    with torch.no_grad():
        velocity = model(latents, torch.tensor([700.0, 300.0]), text, mask, skip_layers)

    golden = json.loads(VELOCITIES.read_text(encoding="utf-8"))[expected]
    assert torch.equal(velocity, torch.tensor(golden, dtype=torch.float32))  # bit for bit, as before references


def test_moss_transformer_reads_a_references_content_into_the_drawn_images_velocity_not_its_shape() -> None:
    model = _seeded_dit()
    latents, text, mask = _inputs()
    reference, other = torch.randn(2, 2, 4, 2, 6, generator=torch.Generator().manual_seed(3))  # a grid of its own size
    timesteps = torch.tensor([700.0, 300.0])
    with torch.no_grad():
        alone = model(latents, timesteps, text, mask)
        from_other = model(latents, timesteps, text, mask, references=[other])

        referenced = model(latents, timesteps, text, mask, references=[reference])

    assert referenced.shape == alone.shape == (2, 4, 3, 5)
    assert not torch.allclose(referenced, alone, rtol=0, atol=1e-4)
    assert not torch.allclose(referenced, from_other, rtol=0, atol=1e-4)


def _block_modulation(model: MossTransformer, *args: object, **kwargs: object) -> torch.Tensor:
    """The time modulation the first joint block is given in a forward of ``model``."""
    seen: list[torch.Tensor] = []

    def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen.append(inputs[2])

    hook = model.layers[0].register_forward_pre_hook(capture)
    with torch.no_grad():
        model(*args, **kwargs)
    hook.remove()
    return seen[0]


def test_moss_transformer_modulates_reference_tokens_as_at_timestep_0_and_the_rest_at_the_flow_timestep() -> None:
    model = _seeded_dit()
    latents, text, mask = _inputs()
    latents, text, mask = latents[:1], text[:1], mask[:1]
    reference = torch.randn(1, 4, 2, 2, generator=torch.Generator().manual_seed(3))
    at_t = _block_modulation(model, latents, torch.tensor([700.0]), text, mask)  # shared by every token: [1, 6, 32]
    at_0 = _block_modulation(model, latents, torch.tensor([0.0]), text, mask)

    by_token = _block_modulation(model, latents, torch.tensor([700.0]), text, mask, references=[reference])

    drawn = 3 * 5 + 6  # the drawn image's 15 tokens, then the prompt's 6; then the reference's 4
    assert by_token.shape == (1, drawn + 4, 6, 32)
    assert torch.equal(by_token[:, :drawn], at_t[:, None].expand(1, drawn, 6, 32))
    assert torch.equal(by_token[:, drawn:], at_0[:, None].expand(1, 4, 6, 32))


def test_token_positions_put_reference_k_on_the_reference_axis_at_k_plus_1_centred_on_its_own_grid() -> None:
    positions = token_positions(2, 2, 3, torch.device("cpu"), references=[(1, 2), (2, 1)])

    drawn = [[0, 3, -0.5, -0.5], [0, 3, -0.5, 0.5], [0, 3, 0.5, -0.5], [0, 3, 0.5, 0.5]]
    prompt = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]
    references = [[1, 3, 0, -0.5], [1, 3, 0, 0.5], [2, 3, -0.5, 0], [2, 3, 0.5, 0]]
    assert positions.tolist() == drawn + prompt + references


@pytest.fixture(scope="module")
def text_encoder_pipe(tiny_moss: Path) -> ModularPipeline:
    """The tiny MOSS's text encoder step alone, its text encoder and tokenizer loaded; no guider, so no negative."""
    blocks = SequentialPipelineBlocks.from_blocks_dict({"text_encoder": MossTextEncoderStep()})
    pipe = MossModularPipeline(blocks=blocks, pretrained_model_name_or_path=str(tiny_moss))
    pipe.load_components(dtype=torch.float32)
    return pipe


def _prompt_of(tokens: int) -> str:
    """A prompt MOSS's text encoder reads as ``tokens`` tokens, its chat template's 10 included
    (``<|im_start|>user\\n``, then ``<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``): "ord ord ...", a token a word."""
    return " ".join(["ord"] * (tokens - 10))


def test_moss_text_encoder_encodes_a_prompt_of_4096_tokens_whole(text_encoder_pipe: ModularPipeline) -> None:
    embeds = text_encoder_pipe(prompt=_prompt_of(4096), output="prompt_embeds")

    assert embeds.shape == (1, 4096, 16)


def test_moss_text_encoder_refuses_a_prompt_one_token_past_4096(text_encoder_pipe: ModularPipeline) -> None:
    prompt = _prompt_of(4097)

    with pytest.raises(ValueError, match="the prompt is 4097 tokens, past the text encoder's 4096"):
        text_encoder_pipe(prompt=prompt, output="prompt_embeds")


def test_moss_repo_draws_with_the_transformer_folder_it_names(tiny_moss: Path, tmp_path: Path) -> None:
    base = Path(ModelIndex.read(tiny_moss).source("text_encoder").repo)
    torch.manual_seed(1)
    trained = MossTransformer.from_config(MossTransformer.load_config(base / "transformer"))
    trained.save_pretrained(tmp_path / "export")

    write_moss_repo(tmp_path / "model", base=base, transformer=tmp_path / "export")

    pipe = MossModularPipeline.from_pretrained(str(tmp_path / "model"))
    pipe.load_components(names=["transformer"])
    assert torch.equal(pipe.transformer.final_linear_video.weight, trained.final_linear_video.weight)
