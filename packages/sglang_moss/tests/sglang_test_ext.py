"""diffu_sglang_moss as the tests load it into SGLang when a test asks more of it (``SGLANG_EXTERNAL_MODEL_PACKAGE``
names this module, which imports the plugin, and the plugin registers itself), by switches in the environment:

- ``DIFFU_TEST_ALLOW_COMPILE=1``: SGLang's refusal of torch.compile for MOSS-Image lifted, its other checks kept (as
  the bench extension ``runs/htr/moss_bench/moss_bench_ext`` does), and Inductor told to round where eager rounds
  (``emulate_precision_casts``), so a compiled block stays within float noise of the eager one (its reductions still
  run in Inductor's order: not bit for bit).
- ``DIFFU_TEST_RECORD=<file>``: one JSON line a call appended to the file for each of SGLang's fused helpers the
  plugin's joint block calls (``_norm_modulate``, ``residual_gate_add``) and for the joint Q/K/V weight
  (``qkv_weight``): its name, its tensors' shapes and its result's; and one a call of a quantized projection
  (:mod:`diffu_sglang_moss.precision`): its class and which layer (``id``) it is.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from sglang.multimodal_gen.runtime.server_args import ServerArgs

import diffu_sglang_moss  # noqa: F401  registers the pipeline
from diffu_sglang_moss import dit, precision
from diffu_sglang_moss.pipeline import DiffuMOSSImagePipelineConfig


def _validate(self: DiffuMOSSImagePipelineConfig, server_args: ServerArgs) -> None:
    if server_args.tp_size != 1:
        raise ValueError("MOSSImage supports sequence and CFG parallelism, not TP.")
    if not server_args.disable_autocast:
        raise ValueError("MOSSImage requires disabled autocast to preserve reference precision.")


def _recorded(name: str, helper: Callable[..., Any], record: Path) -> Callable[..., Any]:
    def call(*args: Any) -> Any:
        result = helper(*args)
        shapes = [list(arg.shape) for arg in args if isinstance(arg, torch.Tensor)]
        line = {"helper": name, "shapes": shapes, "result": list(result.shape)}
        with record.open("a", encoding="utf-8") as lines:
            lines.write(json.dumps(line) + "\n")
        return result

    return call


def _recorded_forward(layer_class: type[precision.QuantizedLinear], record: Path) -> None:
    forward = layer_class.forward

    def call(self: precision.QuantizedLinear, x: torch.Tensor) -> torch.Tensor:
        with record.open("a", encoding="utf-8") as lines:
            lines.write(json.dumps({"helper": layer_class.__name__, "layer": id(self)}) + "\n")
        return forward(self, x)

    layer_class.forward = call


if os.environ.get("DIFFU_TEST_ALLOW_COMPILE") == "1":
    import torch._inductor.config as inductor_config

    DiffuMOSSImagePipelineConfig.validate_server_args = _validate
    inductor_config.emulate_precision_casts = True

if record := os.environ.get("DIFFU_TEST_RECORD"):
    dit._norm_modulate = _recorded("_norm_modulate", dit._norm_modulate, Path(record))
    dit.residual_gate_add = _recorded("residual_gate_add", dit.residual_gate_add, Path(record))
    dit.qkv_weight = _recorded("qkv_weight", dit.qkv_weight, Path(record))
    for quantized in (precision.MXFP8Linear, precision.RowwiseFP8Linear):
        _recorded_forward(quantized, Path(record))
