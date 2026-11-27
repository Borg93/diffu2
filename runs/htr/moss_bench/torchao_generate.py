"""diffu-generate with the DiT's joint-block projections quantized by torchao (MXFP8 or NVFP4) before they compile.

PyTorch's recipe for Blackwell (pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao):
dynamic activation and weight quantization of the large Linear layers, then the repeated blocks compiled. The layers
are the ones our own MXFP8 swaps (``diffu_moss.mxfp8.DIT_LINEARS``: q, k, v, o and the feed-forward's two matrices of
every joint block, all 3,072 wide or more; embeddings, norms and the head stay bfloat16). TORCHAO names the format:
``none`` (bfloat16, this environment's baseline), ``mxfp8``, ``nvfp4``. Give draw settings with ``compile_blocks``.

    TORCHAO=nvfp4 python torchao_generate.py <diffu-generate's arguments>
"""

import os

import torch
from torch import nn

import diffu_page.draw.pipeline as pipeline
from diffu_moss.mxfp8 import DIT_LINEARS
from diffu_page.cli.generate import app

FORMAT = os.environ.get("TORCHAO", "none")
prepare_of = pipeline.prepare_inference


def config():
    from torchao.prototype.mx_formats.inference_workflow import (
        MXDynamicActivationMXWeightConfig,
        NVFP4DynamicActivationNVFP4WeightConfig,
    )
    from torchao.quantization.quantize_.common import KernelPreference

    if FORMAT == "mxfp8":
        return MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn, weight_dtype=torch.float8_e4m3fn, kernel_preference=KernelPreference.AUTO
        )
    if FORMAT == "nvfp4":
        return NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True, use_triton_kernel=True)
    raise SystemExit(f"TORCHAO is none, mxfp8 or nvfp4, not {FORMAT!r}")


def prepare(pipe, options, *, steps):
    if FORMAT != "none":
        from torchao.quantization import quantize_

        chosen = [name for name, module in pipe.transformer.named_modules() if isinstance(module, nn.Linear) and DIT_LINEARS.fullmatch(name)]
        quantize_(pipe.transformer, config(), filter_fn=lambda module, name: isinstance(module, nn.Linear) and bool(DIT_LINEARS.fullmatch(name)))
        torch.cuda.empty_cache()
        print(f"torchao {FORMAT}: {len(chosen)} joint-block projections quantized", flush=True)
    prepare_of(pipe, options, steps=steps)


pipeline.prepare_inference = prepare

if __name__ == "__main__":
    app()
