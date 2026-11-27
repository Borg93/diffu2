"""Item B on the GPU: the DiT's joint-block projections in MXFP8 or row-wise FP8 (the plugin's
``DIFFU_SGLANG_DIT_PRECISION``, :mod:`diffu_sglang_moss.precision`) against bfloat16: how far one block's output is
from the bfloat16 block's, and how fast one block and one page's DiT pass (42 blocks, a 2 MP page with its sample:
15,700 tokens, dim 3072, 24 heads) run, eager and compiled as regional compile compiles them (``JointLayer.compile``
with SGLang's arguments). Before that, the pieces: one projection's GEMM alone (its input quantized before), and the
input's quantization alone (plain ops; the compiled function the plugin runs outside a compiled region; SGLang's own
Triton MXFP8 quantizer, which rounds its scale up rather than by the OCP spec, for comparison).

Run on GPU 2, quiet, in the SGLang environment (the compiles take minutes under max-autotune):

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=packages/sglang_moss/src:runs/htr/moss_bench \
        /mnt/scratch/gabriel/envs/sglang_moss/bin/python runs/htr/moss_bench/check_b_quantized_dit.py

PRECISIONS=bf16,mxfp8 leaves row-wise FP8 out; SKIP_PAGE=1 times the block alone. The block's error is given against
its update (output minus input), since the residual it adds to is not quantized. The weights are random (the norms'
scales about 1): the error of the trained model's weights can differ; the page benchmark (sglang_bench.py) judges that.
"""

import copy
import os
import time

import torch
from gpu_block import LAYERS, block, inputs, per_page, timed
from torch import nn

from diffu_sglang_moss.dit import JointLayer
from diffu_sglang_moss.precision import (
    DitPrecision,
    MXFP8Linear,
    RowwiseFP8Linear,
    fp8_rows,
    mx_rows,
    quantize_projections,
    quantized_input,
)

MODE = os.environ.get("SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")  # SGLang's default
PRECISIONS = [DitPrecision(name) for name in os.environ.get("PRECISIONS", "bf16,mxfp8,fp8_rowwise").split(",")]


def quantized(layer: nn.Module, precision: DitPrecision) -> nn.Module:
    """A copy of the block with its projections in ``precision``, as the plugin swaps them."""
    held = nn.Module()
    held.layers = nn.ModuleList([copy.deepcopy(layer)])
    quantize_projections(held, precision)
    return held.layers[0]


def pieces() -> None:
    """One projection's GEMM (ffn_up_gate, 3072 -> 16384, and q, 3072 -> 3072) and the input's quantization."""
    torch.manual_seed(0)
    rows = torch.randn(15700, 3072, device="cuda", dtype=torch.bfloat16)
    for name, out_features in (("q", 3072), ("ffn_up_gate", 16384)):
        linear = nn.Linear(3072, out_features, bias=False, device="cuda", dtype=torch.bfloat16)
        mx, fp8 = MXFP8Linear.from_linear(linear), RowwiseFP8Linear.from_linear(linear)
        a_mx, s_mx = mx_rows(rows)
        a_fp8, s_fp8 = fp8_rows(rows)
        runs = {
            "bf16": lambda linear=linear: linear(rows),
            "mxfp8": lambda mx=mx, a=a_mx, s=s_mx: torch.nn.functional.scaled_mm(
                a,
                mx.elements.view(torch.float8_e4m3fn).t(),
                s,
                torch.nn.functional.ScalingType.BlockWise1x32,
                mx.scales.view(torch.float8_e8m0fnu),
                torch.nn.functional.ScalingType.BlockWise1x32,
                swizzle_a=torch.nn.functional.SwizzleType.SWIZZLE_32_4_4,
                swizzle_b=torch.nn.functional.SwizzleType.SWIZZLE_32_4_4,
                output_dtype=torch.bfloat16,
            ),
            "fp8_rowwise": lambda fp8=fp8, a=a_fp8, s=s_fp8: torch._scaled_mm(
                a,
                fp8.elements.view(torch.float8_e4m3fn).t(),
                scale_a=s,
                scale_b=fp8.scales.view(torch.float32),
                out_dtype=torch.bfloat16,
            ),
        }
        reference = runs["bf16"]().float()
        for precision, run in runs.items():
            try:
                ms, out = timed(run)
            except RuntimeError as error:  # a recipe this card or this torch does not run
                print(f"GEMM {name:11s} {precision:11s} FAILED: {str(error)[:200]}")
                continue
            error = (out.float() - reference).norm() / reference.norm()
            print(f"GEMM {name:11s} {precision:11s} {ms:7.3f} ms  rel-RMS vs bf16 {error:.3e}")
    for precision, quantizer in (("mxfp8", mx_rows), ("fp8_rowwise", fp8_rows)):
        ms, _ = timed(lambda quantizer=quantizer: quantizer(rows))
        print(f"input quantization {precision:11s} plain ops      {ms:7.3f} ms")
        ms, _ = timed(lambda quantizer=quantizer: quantized_input(quantizer, rows))
        print(f"input quantization {precision:11s} compiled       {ms:7.3f} ms")
    try:
        from sglang.kernels.ops.diffusion import mxfp8_quantize_swizzled

        ms, _ = timed(lambda: mxfp8_quantize_swizzled(rows))
        print(f"input quantization {'mxfp8':11s} SGLang Triton  {ms:7.3f} ms  (scale rounded up: another rule)")
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"SGLang's Triton MXFP8 quantizer FAILED: {str(error)[:200]}")


def blocks(precision: DitPrecision, layers: list[nn.Module], compiled: bool) -> list[JointLayer]:
    joint = [
        JointLayer(layer if precision is DitPrecision.BF16 else quantized(layer, precision), k)
        for k, layer in enumerate(layers)
    ]
    if compiled:
        for module in joint:
            module.compile(fullgraph=False, dynamic=None, mode=MODE)  # SGLang's build_torch_compile_kwargs
    return joint


def main() -> None:
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.__version__, MODE)
    with torch.no_grad():
        pieces()
        count = 1 if os.environ.get("SKIP_PAGE") == "1" else LAYERS
        layers = [block(seed) for seed in range(count)]
        x, rope, modulation = inputs()
        reference = JointLayer(layers[0], 0)(x, rope, modulation, frozenset()).float()
        update = (reference - x.float()).norm()
        for compiled in (False, True):
            for precision in PRECISIONS:
                label = f"{precision:11s} {'compiled' if compiled else 'eager':8s}"
                try:
                    joint = blocks(precision, layers, compiled)
                    began = time.time()
                    out = joint[0](x, rope, modulation, frozenset())
                    torch.cuda.synchronize()
                    first = time.time() - began
                except RuntimeError as error:
                    print(f"{label} FAILED: {str(error)[:300]}")
                    continue
                error = (out.float() - reference).norm()
                print(
                    f"{label} block: first call {first:6.1f} s; vs bf16 eager rel-RMS {error / reference.norm():.3e}"
                    f" of the output, {error / update:.3e} of the update"
                )
                ms, _ = timed(lambda joint=joint: joint[0](x, rope, modulation, frozenset()))
                print(f"{label} block {ms:8.3f} ms   {per_page(ms):6.1f} s a page (42 blocks, 2 passes, 28 steps)")
                if count > 1:

                    def page(joint: list[JointLayer] = joint) -> torch.Tensor:
                        hidden = x
                        for module in joint:
                            hidden = module(hidden, rope, modulation, frozenset())
                        return hidden

                    page()  # the other blocks' first calls
                    ms, _ = timed(page, iters=3, warmup=1)
                    print(f"{label} DiT pass {ms:8.1f} ms   {ms * 2 * 28 / 1000:6.1f} s a page (2 passes, 28 steps)")
                del joint
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
