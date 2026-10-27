"""MOSS-Image on CPU: our DiT predicts the velocity the verified implementation predicted, the text encoder refuses a
prompt past its 4,096 tokens, and a MOSS repo draws with the DiT folder it names."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import pytest
import torch
from diffusers.modular_pipelines import ModularPipeline, SequentialPipelineBlocks

from diffu.repo import ModelIndex
from diffu_moss.init import write_moss_repo
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.modular_pipelines.moss.encoders import MossTextEncoderStep
from diffu_moss.transformer import MossTransformer

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


@pytest.mark.parametrize(
    ("skip_layers", "expected"),
    [
        pytest.param(frozenset(), "velocity", id="every-block"),
        pytest.param(frozenset({1}), "velocity_skipping_1", id="block-1-skipped"),
    ],
)
def test_moss_transformer_predicts_the_velocity_the_verified_implementation_predicted(
    skip_layers: frozenset[int], expected: str
) -> None:
    model = _seeded_dit()
    latents = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(1))  # a grid of 3 rows, 5 columns
    text = torch.randn(2, 6, 8, generator=torch.Generator().manual_seed(2))
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)  # the second prompt 3 tokens long

    with torch.no_grad():
        velocity = model(latents, torch.tensor([700.0, 300.0]), text, mask, skip_layers)

    golden = json.loads(VELOCITIES.read_text(encoding="utf-8"))[expected]
    torch.testing.assert_close(velocity, torch.tensor(golden, dtype=torch.float32))


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
