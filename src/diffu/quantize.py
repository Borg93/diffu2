"""Post-training weight quantization of the SD3 backbone for faster inference (torchao).

Weight-only quantization of the MMDiT transformer's Linear layers — the bulk of inference compute
(``sample_steps`` denoise iterations × CFG). The frozen VAE + style encoder run O(1) per generation,
so they stay bf16. Quantization is POST-TRAINING and INFERENCE-ONLY — it never touches training.

The speedup comes from compiled low-bit matmul kernels, so quantize BEFORE ``Backbone.compile_blocks``
(torchao + ``torch.compile`` is the supported combo; int8 in eager dequantizes and can be SLOWER).
bf16 is the accuracy baseline; int8/fp8 trade a little precision for speed/memory on memory-bound
GEMMs — always validate gen_CER before shipping (``scripts/bench_quant.py``).
"""

from __future__ import annotations

from typing import Literal

import torch

QuantMode = Literal["int8", "int8dq", "fp8", "fp8dq"]


def quantize_backbone(model: torch.nn.Module, mode: QuantMode = "int8") -> None:
    """In-place torchao weight quantization of ``model.backbone.transformer`` Linear layers.

    Args:
        model: a built ``Diffu`` (needs ``.backbone.transformer``).
        mode: ``int8`` = int8 weight-only; ``int8dq`` = int8 dynamic-activation + int8 weight;
            ``fp8`` = float8 weight-only; ``fp8dq`` = float8 dynamic-activation + float8 weight
            (full fp8 GEMM — best on Blackwell's fp8 tensor cores). fp8 modes need CC ≥ 8.9.

    Raises:
        ValueError: unknown ``mode``.
        RuntimeError: an fp8 mode requested on pre-Ada hardware.
    """
    from torchao.quantization import (  # torchao 0.17 config-based API
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )

    # int8 version=2: the maintained Int8Tensor path (torchao 0.17 deprecates the v1 default's legacy
    # AffineQuantizedTensor route). set_inductor_config=False everywhere: the default True flips GLOBAL
    # inductor flags + TF32 at quantize time, which makes any quant-vs-baseline benchmark
    # apples-to-oranges — callers own their global compile settings.
    recipes = {
        "int8": lambda: Int8WeightOnlyConfig(version=2, set_inductor_config=False),
        "int8dq": lambda: Int8DynamicActivationInt8WeightConfig(version=2, set_inductor_config=False),
        "fp8": lambda: Float8WeightOnlyConfig(set_inductor_config=False),
        "fp8dq": lambda: Float8DynamicActivationFloat8WeightConfig(set_inductor_config=False),
    }
    if mode not in recipes:
        raise ValueError(f"unknown quant mode {mode!r}; choose from {sorted(recipes)}")
    if mode.startswith("fp8") and torch.cuda.get_device_capability() < (8, 9):
        raise RuntimeError("fp8 needs compute capability >= 8.9 (Ada/Hopper/Blackwell)")
    quantize_(model.backbone.transformer, recipes[mode]())
