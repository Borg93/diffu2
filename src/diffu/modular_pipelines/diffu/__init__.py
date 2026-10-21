"""diffu as a Modular Diffusers pipeline, laid out as diffusers lays out a model's modular pipeline."""

from diffu.modular_pipelines.diffu.modular_blocks_diffu import DiffuAutoBlocks
from diffu.modular_pipelines.diffu.modular_pipeline import DiffuModularPipeline

__all__ = ["DiffuAutoBlocks", "DiffuModularPipeline"]
