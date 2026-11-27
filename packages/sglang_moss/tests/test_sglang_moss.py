"""SGLang Diffusion with diffu_sglang_moss draws, on CPU, bit for bit the image our own pipeline draws from the same
model, prompt and noise, with a sample page (over the page, or above it for a DiT trained so) and without one, at
both pipelines' default guidance; under Cache-DiT, the same image when no step is skipped and another when steps are;
and a model whose DiT takes no reference is refused one. The joint block hands SGLang's fused helpers each part of
the tokens with one modulation row a sample; with our guidance stop SGLang draws what our pipeline draws with it;
with Q/K/V as one GEMM, the same image within rounding; under regional compile, the joint blocks alone are compiled
and draw the same image within float noise.

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
from diffusers.guiders import ClassifierFreeGuidance
from PIL import Image

from diffu_moss.export_hf import export_hf
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.modular_pipelines.moss.denoise import GUIDANCE_SCALE

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


def _ours(model: Path, sample: Path | None, guidance_stop: float = 1.0) -> np.ndarray:
    """Our pipeline's image ``[3, H, W]`` in [0, 1]: the VAE in float32, the rest in bfloat16, as the page pipeline
    loads it and as SGLang's MOSS-Image config does; guided until ``guidance_stop`` of the steps, as the page
    pipeline's guider is (its ``stop``)."""
    pipe = MossModularPipeline.from_pretrained(str(model))
    pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    if guidance_stop != 1.0:
        pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=GUIDANCE_SCALE, stop=guidance_stop))
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


def _sglang(
    model: Path,
    sample: Path | None,
    work: Path,
    cache: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
    server_extra: dict[str, object] | None = None,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """SGLang drawing from ``model`` on CPU, its image saved to ``work / "sglang.npy"``; with ``cache``, under
    Cache-DiT with those settings (a request's ``cache_dit_params``); ``extra`` more sampling parameters,
    ``server_extra`` more server arguments; ``env_extra`` set in SGLang's environment, which loads the test extension
    (``sglang_test_ext``, the plugin and what a test asks of it) when it names its switches."""
    params: dict[str, object] = {"prompt": PROMPT, "height": SIZE, "width": SIZE, "num_inference_steps": STEPS}
    params |= {"seed": SEED, "generator_device": "cpu", "uncond_skip_layers": UNCOND_SKIP_LAYERS}
    if sample is not None:
        params["image_path"] = str(sample)
    if cache is not None:
        params |= {"enable_cache_dit": True, "cache_dit_params": cache}
    params |= extra or {}
    server = {"num_gpus": 1, "attention_backend": "torch_sdpa", "disable_autocast": True, **(server_extra or {})}
    request = work / "request.json"
    request.write_text(json.dumps({"model": str(model), "params": params, "server": server}), encoding="utf-8")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "cpu",
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "sglang_test_ext" if env_extra else "diffu_sglang_moss",
        "PYTHONPATH": os.pathsep.join((str(PLUGIN), str(Path(__file__).parent))),
        "HF_HUB_OFFLINE": "1",
        **(env_extra or {}),
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


def test_sglang_lays_the_sample_above_the_page_for_a_dit_trained_so(
    tiny_moss_reference_above: Path, tmp_path: Path
) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png")
    ours = _ours(tiny_moss_reference_above, sample)

    drawn = _sglang(tmp_path / "hf", sample, tmp_path)

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    np.testing.assert_array_equal(np.load(tmp_path / "sglang.npy"), ours)


NEVER_SKIPS = {"residual_diff_threshold": 0.0, "max_warmup_steps": 0}
SKIPS_WHEN_IT_MAY = {"residual_diff_threshold": 1e6, "max_warmup_steps": 0, "max_continuous_cached_steps": STEPS}


@pytest.mark.parametrize("with_sample", [False, True], ids=["text-only", "sample-page"])
def test_cache_dit_that_skips_no_step_draws_what_our_pipeline_draws(
    tiny_moss_reference_above: Path, tmp_path: Path, with_sample: bool
) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png") if with_sample else None
    ours = _ours(tiny_moss_reference_above, sample)

    drawn = _sglang(tmp_path / "hf", sample, tmp_path, cache=NEVER_SKIPS)

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    assert "Enabling cache-dit" in drawn.stdout + drawn.stderr
    np.testing.assert_array_equal(np.load(tmp_path / "sglang.npy"), ours)


def test_cache_dit_skips_steps_of_a_page_drawn_from_a_sample(tiny_moss_reference_above: Path, tmp_path: Path) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png")
    ours = _ours(tiny_moss_reference_above, sample)

    drawn = _sglang(tmp_path / "hf", sample, tmp_path, cache=SKIPS_WHEN_IT_MAY)

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    cached = np.load(tmp_path / "sglang.npy")
    assert cached.shape == ours.shape
    assert np.isfinite(cached).all()
    assert not np.array_equal(cached, ours)  # steps were skipped: the image is another


def test_sglang_refuses_a_sample_to_a_model_whose_dit_takes_none(tiny_moss: Path, tmp_path: Path) -> None:
    export_hf(tiny_moss, tmp_path / "hf")

    drawn = _sglang(tmp_path / "hf", _sample(tmp_path / "sample.png"), tmp_path)

    assert drawn.returncode != 0
    assert "this MOSS-Image model takes no reference image" in drawn.stdout + drawn.stderr


GUIDANCE_STOP = 0.5  # of 3 steps: the first guided, the last two the conditional pass alone


def test_sglang_stops_guidance_where_our_pipeline_does(tiny_moss_reference_above: Path, tmp_path: Path) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png")
    ours = _ours(tiny_moss_reference_above, sample, guidance_stop=GUIDANCE_STOP)
    assert not np.array_equal(ours, _ours(tiny_moss_reference_above, sample))  # the stop changes the page

    drawn = _sglang(tmp_path / "hf", sample, tmp_path, extra={"guidance_stop": GUIDANCE_STOP})

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    np.testing.assert_array_equal(np.load(tmp_path / "sglang.npy"), ours)


def _recorded(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_joint_block_hands_sglangs_fused_helpers_one_modulation_row_a_part(
    tiny_moss_reference_above: Path, tmp_path: Path
) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png")
    record = tmp_path / "calls.jsonl"

    drawn = _sglang(tmp_path / "hf", sample, tmp_path, env_extra={"DIFFU_TEST_RECORD": str(record)})

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    calls = _recorded(record)
    reference_tokens = (SAMPLE_SIZE[0] // 16) * (SAMPLE_SIZE[1] // 16)
    for helper, rows in (("_norm_modulate", [[1, 32], [1, 32]]), ("residual_gate_add", [[1, 1, 32]])):
        parts = [call["shapes"] for call in calls if call["helper"] == helper]
        assert parts, f"{helper} was never called"
        assert all(shapes[-len(rows) :] == rows for shapes in parts)  # its scale and shift, or its gate: one row
        assert {shapes[0][1] for shapes in parts[1::2]} == {reference_tokens}  # every second part: the sample's


def test_qkv_as_one_gemm_draws_what_three_draw_within_rounding(tiny_moss_reference_above: Path, tmp_path: Path) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png")
    ours = _ours(tiny_moss_reference_above, sample)
    record = tmp_path / "calls.jsonl"

    drawn = _sglang(
        tmp_path / "hf", sample, tmp_path, extra={"fused_qkv": True}, env_extra={"DIFFU_TEST_RECORD": str(record)}
    )

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    assert any(call["helper"] == "qkv_weight" and call["result"] == [96, 32] for call in _recorded(record))
    np.testing.assert_allclose(np.load(tmp_path / "sglang.npy"), ours, atol=2e-2)


REGIONAL_COMPILE = {"enable_torch_compile": True, "regional_compile": True}


@pytest.mark.parametrize("with_sample", [False, True], ids=["text-only", "sample-page"])
def test_regional_compile_compiles_the_joint_blocks_alone(
    tiny_moss_reference_above: Path, tmp_path: Path, with_sample: bool
) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png") if with_sample else None
    ours = _ours(tiny_moss_reference_above, sample)

    drawn = _sglang(
        tmp_path / "hf",
        sample,
        tmp_path,
        server_extra=REGIONAL_COMPILE,
        env_extra={"DIFFU_TEST_ALLOW_COMPILE": "1", "SGLANG_TORCH_COMPILE_MODE": "default"},
    )

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    layers = json.loads((tmp_path / "hf" / "transformer" / "config.json").read_text(encoding="utf-8"))["num_layers"]
    assert f"Enabled regional torch.compile for {2 * layers} submodules" in drawn.stdout + drawn.stderr
    # Inductor reduces in its own order: the page within float noise of the eager one (2.7e-6 measured), not its bits
    np.testing.assert_allclose(np.load(tmp_path / "sglang.npy"), ours, rtol=0, atol=1e-4)
