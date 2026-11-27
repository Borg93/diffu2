"""MOSS-Image in SGLang Diffusion, drawing from a reference image (a sample page) as diffu_moss does.

SGLang imports this package in every process it starts when ``SGLANG_EXTERNAL_MODEL_PACKAGE=diffu_sglang_moss``;
importing it registers the reference-reading DiT (``DiffuMOSSImageTransformer``) and puts our pipeline in the place of
SGLang's ``MOSSImagePipeline``, its config and sampling parameters (diffu_moss's guidance defaults) chosen for every
local MOSS-Image folder. A request without a reference runs SGLang's own stages and DiT.
"""

from __future__ import annotations

from sglang.multimodal_gen.registry import register_pipeline
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

from diffu_sglang_moss.pipeline import (
    TRANSFORMER,
    DiffuMOSSImagePipeline,
    DiffuMOSSImagePipelineConfig,
    DiffuMOSSImageSamplingParams,
)


def _is_moss_image(probe: str) -> bool:
    """Whether a model path or pipeline class names MOSS-Image, as SGLang's own detector reads it."""
    return "mossimage" in probe.lower().replace("-", "").replace("_", "")


try:  # imported here, before any model is built: SGLang 0.5.23 imports it first while building the DiT on the meta
    # device, where its module-level `(torch.ones(()) / 448.0).item()` fails and the worker exits (fp8_fa_sm120)
    import sglang.kernels.ops.attention.fp8_fa_sm120  # noqa: F401
except Exception:  # noqa: BLE001  not every build or GPU has the kernel; only that backend needs it
    pass

ModelRegistry.register_model(TRANSFORMER, f"diffu_sglang_moss.dit:{TRANSFORMER}")
register_pipeline(
    DiffuMOSSImagePipeline,
    sampling_param_cls=DiffuMOSSImageSamplingParams,
    pipeline_config_cls=DiffuMOSSImagePipelineConfig,
    model_detectors=[_is_moss_image],
    overwrite=True,
)
