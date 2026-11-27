"""SGLang Diffusion with diffu_sglang_moss draws, on CPU, bit for bit the image our own pipeline draws from the same
model, prompt and noise, with a sample page (over the page, or above it for a DiT trained so) and without one, at
both pipelines' default guidance; under Cache-DiT, the same image when no step is skipped and another when steps are;
and a model whose DiT takes no reference is refused one. The joint block hands SGLang's fused helpers each part of
the tokens with one modulation row a sample; with our guidance stop SGLang draws what our pipeline draws with it;
with Q/K/V as one GEMM, the same image within rounding; under regional compile, the joint blocks alone are compiled
and draw the same image within float noise. The DiT's projection precision (``DIFFU_SGLANG_DIT_PRECISION``): at
``bf16`` the image is the same bit for bit; in MXFP8 every joint-block projection is swapped and SGLang draws what our
pipeline draws with its own MXFP8, eager and under regional compile; in row-wise FP8 they are swapped and the image
stays near the bfloat16 one; ``fused_qkv`` on a quantized DiT and an unknown precision are refused.

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
from diffu_moss.mxfp8 import DIT_LINEARS, quantize_linears

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


def _ours(model: Path, sample: Path | None, guidance_stop: float = 1.0, mxfp8: bool = False) -> np.ndarray:
    """Our pipeline's image ``[3, H, W]`` in [0, 1]: the VAE in float32, the rest in bfloat16, as the page pipeline
    loads it and as SGLang's MOSS-Image config does; guided until ``guidance_stop`` of the steps, as the page
    pipeline's guider is (its ``stop``); with ``mxfp8`` the DiT's joint-block projections in MXFP8, as
    ``MossInference(precision="mxfp8")`` sets them."""
    pipe = MossModularPipeline.from_pretrained(str(model))
    pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    if mxfp8:
        quantize_linears(pipe.transformer, DIT_LINEARS)
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
    (``sglang_test_ext``, the plugin and what a test asks of it) when it names its switches (``DIFFU_TEST_*``)."""
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
        "SGLANG_EXTERNAL_MODEL_PACKAGE": (
            "sglang_test_ext" if any(key.startswith("DIFFU_TEST_") for key in env_extra or {}) else "diffu_sglang_moss"
        ),
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


PRECISION = "DIFFU_SGLANG_DIT_PRECISION"
PROJECTIONS = 6  # a joint block's: q, k, v, o, ffn_up_gate, ffn_down


def _layers(model: Path) -> int:
    return json.loads((model / "transformer" / "config.json").read_text(encoding="utf-8"))["num_layers"]


def test_the_dit_precision_switch_at_bf16_draws_bit_for_bit_as_before(
    tiny_moss_reference_above: Path, tmp_path: Path
) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png")
    ours = _ours(tiny_moss_reference_above, sample)

    drawn = _sglang(tmp_path / "hf", sample, tmp_path, env_extra={PRECISION: "bf16"})

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    assert "layers swapped" not in drawn.stdout + drawn.stderr
    np.testing.assert_array_equal(np.load(tmp_path / "sglang.npy"), ours)


def _swapped(record: Path, layer_class: str) -> int:
    """How many distinct quantized projections of ``layer_class`` ran."""
    return len({call["layer"] for call in _recorded(record) if call["helper"] == layer_class})


@pytest.mark.parametrize("with_sample", [False, True], ids=["text-only", "sample-page"])
def test_an_mxfp8_dit_draws_what_our_pipelines_mxfp8_draws(
    tiny_moss_reference_above: Path, tmp_path: Path, with_sample: bool
) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png") if with_sample else None
    ours = _ours(tiny_moss_reference_above, sample, mxfp8=True)
    assert not np.array_equal(ours, _ours(tiny_moss_reference_above, sample))  # MXFP8 changes the page
    record = tmp_path / "calls.jsonl"

    drawn = _sglang(tmp_path / "hf", sample, tmp_path, env_extra={PRECISION: "mxfp8", "DIFFU_TEST_RECORD": str(record)})

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    layers = PROJECTIONS * _layers(tmp_path / "hf")
    assert f"DiT projections in mxfp8: {layers} layers swapped" in drawn.stdout + drawn.stderr
    assert _swapped(record, "MXFP8Linear") == layers
    # the plugin's MXFP8 is diffu_moss.mxfp8's conversion and CPU emulation, the same operands summed in the same
    # order: the same page bit for bit
    np.testing.assert_array_equal(np.load(tmp_path / "sglang.npy"), ours)


def test_an_fp8_rowwise_dit_swaps_every_projection_and_draws_near_the_bf16_page(
    tiny_moss_reference_above: Path, tmp_path: Path
) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png")
    ours = _ours(tiny_moss_reference_above, sample)
    mxfp8_error = np.abs(_ours(tiny_moss_reference_above, sample, mxfp8=True) - ours).mean()
    record = tmp_path / "calls.jsonl"

    drawn = _sglang(
        tmp_path / "hf", sample, tmp_path, env_extra={PRECISION: "fp8_rowwise", "DIFFU_TEST_RECORD": str(record)}
    )

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    assert _swapped(record, "RowwiseFP8Linear") == PROJECTIONS * _layers(tmp_path / "hf")
    page = np.load(tmp_path / "sglang.npy")
    assert not np.array_equal(page, ours)
    # Its elements are MXFP8's, E4M3 (3 mantissa bits), its scales coarser (one a row or token, not one a block of
    # 32): about MXFP8's distance from the bfloat16 page (whose drawing stays legible on the trained model), at most
    # twice it (measured: 0.0033 against MXFP8's 0.0025, mean absolute in [0, 1])
    assert np.abs(page - ours).mean() < 2 * mxfp8_error


def test_an_mxfp8_dit_under_regional_compile_draws_near_our_pipelines_mxfp8_page(
    tiny_moss_reference_above: Path, tmp_path: Path
) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")
    sample = _sample(tmp_path / "sample.png")
    ours = _ours(tiny_moss_reference_above, sample, mxfp8=True)

    drawn = _sglang(
        tmp_path / "hf",
        sample,
        tmp_path,
        server_extra=REGIONAL_COMPILE,
        env_extra={PRECISION: "mxfp8", "DIFFU_TEST_ALLOW_COMPILE": "1", "SGLANG_TORCH_COMPILE_MODE": "default"},
    )

    assert drawn.returncode == 0, drawn.stdout[-4000:] + drawn.stderr[-4000:]
    log = drawn.stdout + drawn.stderr
    layers = _layers(tmp_path / "hf")
    assert f"DiT projections in mxfp8: {PROJECTIONS * layers} layers swapped" in log
    assert f"Enabled regional torch.compile for {2 * layers} submodules" in log
    # the input's quantization is exact bit arithmetic: the compiled page is off by Inductor's float noise and
    # whatever rounding of an element that noise flips; 2.7e-6 measured, as in bfloat16, so the same bound
    np.testing.assert_allclose(np.load(tmp_path / "sglang.npy"), ours, rtol=0, atol=1e-4)


@pytest.mark.parametrize(
    ("precision", "extra", "refusal"),
    [
        ("mxfp8", {"fused_qkv": True}, "fused_qkv joins the bfloat16 Q, K and V weights"),
        ("int4", {}, "DIFFU_SGLANG_DIT_PRECISION='int4'"),
    ],
    ids=["fused-qkv-on-mxfp8", "unknown-precision"],
)
def test_a_quantized_dit_refuses_fused_qkv_and_an_unknown_precision_is_refused(
    tiny_moss_reference_above: Path, tmp_path: Path, precision: str, extra: dict[str, object], refusal: str
) -> None:
    export_hf(tiny_moss_reference_above, tmp_path / "hf")

    drawn = _sglang(
        tmp_path / "hf", _sample(tmp_path / "sample.png"), tmp_path, extra=extra, env_extra={PRECISION: precision}
    )

    assert drawn.returncode != 0
    assert refusal in drawn.stdout + drawn.stderr
