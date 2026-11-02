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


ModelRegistry.register_model(TRANSFORMER, f"diffu_sglang_moss.dit:{TRANSFORMER}")
register_pipeline(
    DiffuMOSSImagePipeline,
    sampling_param_cls=DiffuMOSSImageSamplingParams,
    pipeline_config_cls=DiffuMOSSImagePipelineConfig,
    model_detectors=[_is_moss_image],
    overwrite=True,
)
