"""Fixtures for diffu's tests: a tiny Qwen3.5 (diffusers' FLUX.3 prompt-upsampler test model) and a tiny diffu repo
around it, with a tiny Qwen-Image VAE and a tiny transformer, all random weights."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLQwenImage, Flux3Transformer2DModel
from huggingface_hub import snapshot_download

from diffu.configs import DiffuConfig, DiffuSize, TextEncoder, Vae
from diffu.init import write_model_repo

TINY_QWEN35 = "hf-internal-testing/tiny-flux3-prompt-upsampler"
TINY_QWEN35_REVISION = "21114f3e05b4f947f6980ad71de1e3ee0f280b0e"  # pinned: a changed fixture must not change results

# 8x downsampling like the real Qwen-Image VAE, so a token is 16 px as in diffu; latent statistics that are not the
# identity, so normalization is exercised
TINY_VAE = {
    "base_dim": 16,
    "z_dim": 4,
    "dim_mult": [1, 1, 1, 1],
    "num_res_blocks": 1,
    "temperal_downsample": [False, True, True],
    "latents_mean": [0.5, -0.5, 0.25, 0.0],
    "latents_std": [2.0, 1.0, 0.5, 1.5],
}


@pytest.fixture(scope="session")
def tiny_qwen35() -> Path:
    """The tiny Qwen3.5 model and tokenizer (a few MB), downloaded once into the Hugging Face cache."""
    return Path(snapshot_download(TINY_QWEN35, revision=TINY_QWEN35_REVISION))


@pytest.fixture(scope="session")
def tiny_repo_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("diffu")


@pytest.fixture(scope="session")
def tiny_config(tiny_qwen35: Path, tiny_repo_dir: Path) -> DiffuConfig:
    """A tiny diffu: 2 heads of 8, one per-mode and one joint block, the tiny Qwen3.5's middle layer, a 4-channel VAE
    kept in the repo."""
    return DiffuConfig(
        size=DiffuSize(num_attention_heads=2, attention_head_dim=8, mode_layers=1, joint_layers=1),
        text_encoder=TextEncoder(repo=str(tiny_qwen35), revision=None, layers=(1,), hidden_size=32),
        vae=Vae(repo=str(tiny_repo_dir), revision=None, subfolder="vae", latent_channels=4),
    )


@pytest.fixture(scope="session")
def tiny_repo(tiny_config: DiffuConfig, tiny_repo_dir: Path) -> Path:
    """The tiny diffu as a model repo. Its transformer keeps its random (not adaLN-Zero) weights, so a prompt or a
    reference changes what it draws."""
    torch.manual_seed(0)
    AutoencoderKLQwenImage(**TINY_VAE).save_pretrained(tiny_repo_dir / "vae")
    transformer = Flux3Transformer2DModel.from_config(tiny_config.transformer_config())
    write_model_repo(transformer, tiny_config, tiny_repo_dir)
    return tiny_repo_dir
