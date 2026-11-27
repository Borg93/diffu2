"""diffu_sglang_moss, with SGLang's refusal of torch.compile for MOSS-Image lifted, to measure it (26 Nov).

SGLang 0.5.23's MOSS-Image config refuses ``enable_torch_compile`` ("support has not been validated"). This package
is given as SGLANG_EXTERNAL_MODEL_PACKAGE in place of the plugin: it imports the plugin (which registers itself) and
drops that one check, keeping the others (TP, autocast). Nothing of SGLang or of the plugin is changed on disk.

With MOSS_BENCH_FA4_SM120=1 (27 Nov) it also lets the "fa" attention backend be FlashAttention 4 on a 12.x card (below).
The plugin's own regional compile conditions (DiffuMOSSImageTransformer._compile_conditions) need nothing here beyond
the lifted refusal: serve with {"enable_torch_compile": true, "regional_compile": true}.
"""

import os

import diffu_sglang_moss  # noqa: F401  registers the pipeline
from diffu_sglang_moss.pipeline import DiffuMOSSImagePipelineConfig


def _validate(self, server_args) -> None:
    if server_args.tp_size != 1:
        raise ValueError("MOSSImage supports sequence and CFG parallelism, not TP.")
    if not server_args.disable_autocast:
        raise ValueError("MOSSImage requires disabled autocast to preserve reference precision.")


DiffuMOSSImagePipelineConfig.validate_server_args = _validate


# 27 Nov: the sample laid above the page, first mended here, is now the plugin's own (reference_place).

# 27 Nov: MOSS_BENCH_FA4_SM120=1 makes the "fa" attention backend FlashAttention 4 on a 12.x card. SGLang 0.5.23 turns
# "fa" into torch SDPA on any 12.x card (platforms/cuda.py, _FlashAttentionBackendResolver), though its vendored FA4
# CuTe code ships an sm120 forward kernel (kernels/ops/attention/fa4_sm120, reached from flash_attn/cute/interface.py
# by get_forward_host). With the switch the resolver answers FA, and SGLang's own FA path then sets version 4 on a
# Blackwell card (_prepare_flash_attention_for_blackwell) and checks the head size. Serve with attention_backend "fa".
if os.environ.get("MOSS_BENCH_FA4_SM120") == "1":
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.platforms.cuda import _FlashAttentionBackendResolver

    def _fa_everywhere(cls, platform) -> AttentionBackendEnum:
        return AttentionBackendEnum.FA

    _FlashAttentionBackendResolver.resolve = classmethod(_fa_everywhere)
