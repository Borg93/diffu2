"""SGLang Diffusion with diffu_sglang_moss draws, on CPU, bit for bit the image our own pipeline draws from the same
model, prompt and noise, with a sample page and without one, at both pipelines' default guidance; and a model whose
DiT takes no reference is refused one.

SGLang runs in its own environment (packages/sglang_moss/README.md): ``DIFFU_SGLANG_PYTHON`` names its interpreter,
which runs ``sglang_draw.py`` with this checkout's plugin first on its path. Without it these tests skip.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from diffu_moss.export_hf import export_hf
from diffu_moss.modular_pipelines.moss import MossModularPipeline

SGLANG_PYTHON = os.environ.get("DIFFU_SGLANG_PYTHON")
pytestmark = pytest.mark.skipif(
    SGLANG_PYTHON is None, reason="DIFFU_SGLANG_PYTHON names no SGLang environment (packages/sglang_moss/README.md)"
)
DRAW = Path(__file__).parent / "sglang_draw.py"
PLUGIN = Path(__file__).parents[1] / "src"
PROMPT = "a page of handwriting"
SIZE, STEPS, SEED = 64, 3, 7
SAMPLE_SIZE = (48, 32)  # px: a 3 x 2 latent grid, not the drawn image's 4 x 4
UNCOND_SKIP_LAYERS = [1]  # the tiny model's (its index's); the plugin's other guidance defaults are ours


def _sample(path: Path) -> Path:
    pixels = np.random.default_rng(3).integers(0, 256, (SAMPLE_SIZE[1], SAMPLE_SIZE[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


def _ours(model: Path, sample: Path | None) -> np.ndarray:
    """Our pipeline's image ``[3, H, W]`` in [0, 1]: the VAE in float32, the rest in bfloat16, as the page pipeline
    loads it and as SGLang's MOSS-Image config does."""
    pipe = MossModularPipeline.from_pretrained(str(model))
    pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    images = pipe(
        prompt=PROMPT,
        height=SIZE,
        width=SIZE,
        num_inference_steps=STEPS,
        generator=torch.Generator("cpu").manual_seed(SEED),
        output_type="pt",
        output="images",
        **({} if sample is None else {"image": Image.open(sample)}),
    )
    return images[0].float().numpy()


def _sglang(model: Path, sample: Path | None, work: Path) -> subprocess.CompletedProcess[str]:
    """SGLang drawing from ``model`` on CPU, its image saved to ``work / "sglang.npy"``."""
    params: dict[str, object] = {"prompt": PROMPT, "height": SIZE, "width": SIZE, "num_inference_steps": STEPS}
    params |= {"seed": SEED, "generator_device": "cpu", "uncond_skip_layers": UNCOND_SKIP_LAYERS}
    if sample is not None:
        params["image_path"] = str(sample)
    server = {"num_gpus": 1, "attention_backend": "torch_sdpa", "disable_autocast": True}
    request = work / "request.json"
    request.write_text(json.dumps({"model": str(model), "params": params, "server": server}), encoding="utf-8")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "cpu",
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "diffu_sglang_moss",
        "PYTHONPATH": str(PLUGIN),
        "HF_HUB_OFFLINE": "1",
    }
    command = [str(SGLANG_PYTHON), str(DRAW), str(request), str(work / "sglang.npy")]
    return subprocess.run(command, env=env, cwd=work, capture_output=True, text=True, timeout=900, check=False)


@pytest.mark.parametrize("with_sample", [False, True], ids=["text-only", "sample-page"])
def test_sglang_draws_bit_for_bit_what_our_pipeline_draws(
    tiny_moss_reference: Path, tmp_path: Path, with_sample: bool
) -> None:
    export_hf(tiny_moss_reference, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png") if with_sample else None
    ours = _ours(tiny_moss_reference, sample)

    drawn = _sglang(tmp_path / "hf", sample, tmp_path)

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    np.testing.assert_array_equal(np.load(tmp_path / "sglang.npy"), ours)


def test_sglang_refuses_a_sample_to_a_model_whose_dit_takes_none(tiny_moss: Path, tmp_path: Path) -> None:
    export_hf(tiny_moss, tmp_path / "hf")

    drawn = _sglang(tmp_path / "hf", _sample(tmp_path / "sample.png"), tmp_path)

    assert drawn.returncode != 0
    assert "this MOSS-Image model takes no reference image" in drawn.stdout + drawn.stderr
