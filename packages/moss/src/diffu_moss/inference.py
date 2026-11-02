"""How MOSS-Image draws fast: the DiT's joint-block projections and the text encoder's layers in bfloat16 or MXFP8
(:mod:`diffu_moss.mxfp8`), the joint blocks run as they are, compiled at load (``torch.compile``, diffusers' regional
compilation) or from an AOTInductor package built ahead of time (:mod:`diffu_moss.aoti`), the sampling grid stepped by
UniPC, which draws as well in fewer steps (:class:`~diffu_moss.modular_pipelines.moss.before_denoise.Sampler`), and a
step cache that skips the joint blocks at some steps (:mod:`diffu_moss.step_cache`). :func:`prepare_inference` applies
them to a loaded pipeline: the page pipeline's ``load_pipeline`` calls it, and so does the benchmark
(:mod:`diffu_moss.bench_inference`). Every option off, the pipeline is left exactly as loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import torch
from pydantic import BaseModel, ConfigDict, model_validator

from diffu_moss.aoti import check_package, install_package, read_package
from diffu_moss.modular_pipelines.moss.before_denoise import Sampler, sampling_sigmas, unipc_sampler
from diffu_moss.mxfp8 import DIT_LINEARS, TEXT_ENCODER_LINEARS, Precision, quantize_linears, supports_mxfp8
from diffu_moss.step_cache import StepCache, enable_step_cache

if TYPE_CHECKING:
    from diffu_moss.modular_pipelines.moss import MossModularPipeline


class MossInference(BaseModel):
    """How a MOSS-Image pipeline computes.

    Attributes:
        precision: The DiT's joint-block projections: their bfloat16 weights, or MXFP8.
        text_encoder_precision: The text encoder's decoder layers (:data:`diffu_moss.mxfp8.TEXT_ENCODER_LINEARS`).
        compile_blocks: Compile the joint blocks at load (regional ``torch.compile``, dynamic shapes); the first page
            pays for the compiling.
        aoti_package: Run the joint blocks from this AOTInductor package (``diffu-moss-aoti``), built for the
            ``precision`` asked, this GPU's architecture and this torch version; it replaces ``compile_blocks``.
        sampler: What steps the sampling grid: MOSS-Image's own Euler sampler, or UniPC.
        cache: A step cache over the joint blocks (:data:`diffu_moss.step_cache.StepCache`), or none.

    Raises:
        ValueError: From validation, if both a package and compiling are asked, or a step cache and compiling.
    """

    model_config = ConfigDict(frozen=True)

    precision: Precision = Precision.BF16
    text_encoder_precision: Precision = Precision.BF16
    compile_blocks: bool = False
    aoti_package: Path | None = None
    sampler: Sampler = Sampler.EULER
    cache: StepCache | None = None

    @model_validator(mode="after")
    def _compiled_one_way(self) -> Self:
        if self.compile_blocks and self.aoti_package is not None:
            raise ValueError("an AOTInductor package's blocks are compiled already: leave compile_blocks off")
        if self.compile_blocks and self.cache is not None:
            raise ValueError(
                "a step cache decides around each joint block, eagerly (torch.compiler.disable), which a block compiled "
                "whole cannot hold: draw with a step cache from an AOTInductor package (aoti_package) or eager blocks"
            )
        return self

    @property
    def quantizes(self) -> bool:
        """Whether any layer computes in MXFP8."""
        return Precision.MXFP8 in (self.precision, self.text_encoder_precision)


def check_inference(options: MossInference, device: torch.device) -> None:
    """What can be refused before the model loads: MXFP8 on a GPU without it, and a package that is not for
    ``device``'s architecture and this process's torch.

    Raises:
        ValueError: If either is asked.
    """
    if options.quantizes and device.type == "cuda" and not supports_mxfp8(device):
        raise ValueError(f"MXFP8 needs a GPU of compute capability 10.0 or later (Blackwell), not {device}")
    if options.aoti_package is not None:
        check_package(read_package(options.aoti_package), device)


def prepare_inference(pipe: MossModularPipeline, options: MossInference, *, steps: int) -> None:
    """``pipe``'s DiT and text encoder (loaded, on their device) set to compute as ``options`` ask, in place, for pages
    of ``steps`` denoising steps (which a step cache may count): the chosen layers quantized to MXFP8 first, the
    scheduler swapped for UniPC, the joint blocks run from the package or compiled, then the step cache's hooks set
    around them.

    Raises:
        ValueError: If the package does not fit the DiT (another precision, width or dtype), or the step cache does
            not fit the passes of guidance (one skips a joint block it needs).
    """
    if options.precision is Precision.MXFP8:
        quantize_linears(pipe.transformer, DIT_LINEARS)
    if options.text_encoder_precision is Precision.MXFP8:
        quantize_linears(pipe.text_encoder, TEXT_ENCODER_LINEARS)
    if options.quantizes and torch.cuda.is_available():
        torch.cuda.empty_cache()  # the bfloat16 weights the quantized layers replaced
    if options.sampler is Sampler.UNIPC:
        pipe.update_components(scheduler=unipc_sampler(pipe.scheduler))
    if options.aoti_package is not None:
        install_package(pipe.transformer, options.aoti_package)
    elif options.compile_blocks:  # dynamic: page sizes and prompt lengths vary, and each new shape would recompile
        pipe.transformer.compile_repeated_blocks(fullgraph=True, dynamic=True)
    if options.cache is not None:  # the grid's levels: fewer than the steps asked when shifted levels share one
        levels = len(sampling_sigmas(steps, pipe.scheduler.config.shift))
        enable_step_cache(pipe.transformer, options.cache, steps=levels, skip_layers=pipe.config.uncond_skip_layers)
