"""Fixtures shared by every workspace member's tests: PAGE pages to plan and draw, diffusers' tiny FLUX.3 test model,
and a tiny diffu model repo (a tiny Qwen3.5, a tiny Qwen-Image VAE and a tiny transformer, all random weights)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLQwenImage, Flux3Transformer2DModel
from huggingface_hub import snapshot_download

from diffu.configs import DiffuConfig, DiffuSize, TextEncoder, Vae
from diffu.init import write_model_repo


def _line(line_id: str, x0: int, x1: int, baseline: int, text: str) -> str:
    return (
        f'<TextLine id="{line_id}"><Coords points="{x0},{baseline - 60} {x1},{baseline - 60} {x1},{baseline + 20} '
        f'{x0},{baseline + 20}"/><Baseline points="{x0},{baseline} {x1},{baseline}"/>'
        f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
    )


@pytest.fixture
def body_and_notes_page(tmp_path: Path) -> Path:
    """A 2000 x 1900 px page: 14 body lines at a 100 px pitch, and two margin notes left of body lines 3 and 4."""
    body = "".join(_line(f"b{i}", 600, 1900, 300 + 100 * i, f"body line {i} of the running text") for i in range(14))
    notes = "".join(_line(f"n{i}", 120, 520, baseline, f"note {i}") for i, baseline in enumerate((600, 700)))
    path = tmp_path / "page.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="page.jpg" imageWidth="2000" imageHeight="1900">'
        f'<TextRegion id="body" type="paragraph">{body}</TextRegion>'
        f'<TextRegion id="notes" type="marginalia">{notes}</TextRegion>'
        "</Page></PcGts>",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def two_column_page(tmp_path: Path) -> Path:
    """A 2500 x 1200 px page with two columns of three lines, planned as two crops of three lines each: the smallest
    page whose crops a pipeline draws more than one of."""
    columns = "".join(
        f'<TextRegion id="{side}" type="paragraph">'
        + "".join(_line(f"{side}{i}", x0, x0 + 1000, 300 + 100 * i, f"{side} line {i} of the text") for i in range(3))
        + "</TextRegion>"
        for side, x0 in (("left", 100), ("right", 1400))
    )
    path = tmp_path / "two_columns.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        f'<Page imageFilename="two_columns.jpg" imageWidth="2500" imageHeight="1200">{columns}</Page></PcGts>',
        encoding="utf-8",
    )
    return path


TINY_FLUX3 = "hf-internal-testing/tiny-flux3-image-modular"
TINY_FLUX3_REVISION = "9130f96f124d6c6fdd1beab63cfb6d2f1c8658a1"  # pinned: a changed fixture must not change results


@pytest.fixture(scope="session")
def tiny_flux3() -> Path:
    """diffusers' tiny FLUX.3 test model (a few MB), downloaded once into the Hugging Face cache."""
    return Path(snapshot_download(TINY_FLUX3, revision=TINY_FLUX3_REVISION))


TINY_QWEN35 = "hf-internal-testing/tiny-flux3-prompt-upsampler"
TINY_QWEN35_REVISION = "21114f3e05b4f947f6980ad71de1e3ee0f280b0e"  # pinned: a changed fixture must not change results

# 8x downsampling like the real Qwen-Image VAE, so a token is 16 px as in diffu; latent statistics that are not the
# identity, so normalization is exercised
TINY_QWEN_IMAGE_VAE = {
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
    """A tiny Qwen3.5 model and tokenizer (diffusers' FLUX.3 prompt-upsampler test model, a few MB)."""
    return Path(snapshot_download(TINY_QWEN35, revision=TINY_QWEN35_REVISION))


@pytest.fixture(scope="session")
def tiny_diffu_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("diffu")


@pytest.fixture(scope="session")
def tiny_diffu_config(tiny_qwen35: Path, tiny_diffu_dir: Path) -> DiffuConfig:
    """A tiny diffu: 2 heads of 8, one per-mode and one joint block, the tiny Qwen3.5's middle layer, a 4-channel VAE
    kept in the repo."""
    return DiffuConfig(
        size=DiffuSize(num_attention_heads=2, attention_head_dim=8, mode_layers=1, joint_layers=1),
        text_encoder=TextEncoder(repo=str(tiny_qwen35), revision=None, layers=(1,), hidden_size=32),
        vae=Vae(repo=str(tiny_diffu_dir), revision=None, subfolder="vae", latent_channels=4),
    )


@pytest.fixture(scope="session")
def tiny_diffu(tiny_diffu_config: DiffuConfig, tiny_diffu_dir: Path) -> Path:
    """The tiny diffu as a model repo. Its transformer keeps its random (not adaLN-Zero) weights, so a prompt or a
    reference changes what it draws."""
    torch.manual_seed(0)
    AutoencoderKLQwenImage(**TINY_QWEN_IMAGE_VAE).save_pretrained(tiny_diffu_dir / "vae")
    transformer = Flux3Transformer2DModel.from_config(tiny_diffu_config.transformer_config())
    write_model_repo(transformer, tiny_diffu_config, tiny_diffu_dir)
    return tiny_diffu_dir
