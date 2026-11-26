"""diffu_sglang_moss, with SGLang's refusal of torch.compile for MOSS-Image lifted, to measure it (26 Nov).

SGLang 0.5.23's MOSS-Image config refuses ``enable_torch_compile`` ("support has not been validated"). This package
is given as SGLANG_EXTERNAL_MODEL_PACKAGE in place of the plugin: it imports the plugin (which registers itself) and
drops that one check, keeping the others (TP, autocast). Nothing of SGLang or of the plugin is changed on disk.
"""

import diffu_sglang_moss  # noqa: F401  registers the pipeline
from diffu_sglang_moss.pipeline import DiffuMOSSImagePipelineConfig


def _validate(self, server_args) -> None:
    if server_args.tp_size != 1:
        raise ValueError("MOSSImage supports sequence and CFG parallelism, not TP.")
    if not server_args.disable_autocast:
        raise ValueError("MOSSImage requires disabled autocast to preserve reference precision.")


DiffuMOSSImagePipelineConfig.validate_server_args = _validate
