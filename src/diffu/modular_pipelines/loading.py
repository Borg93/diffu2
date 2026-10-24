"""Loading a modular pipeline's components: in the dtypes generation and training share, and without letting one that
failed slip through."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from diffusers.modular_pipelines import ModularPipeline

COMPONENT_DTYPES = {"vae": torch.float32, "default": torch.bfloat16}  # load_components' dtype: the VAE in float32


def require_loaded(pipe: ModularPipeline, names: Iterable[str] | None = None) -> None:
    """Fail fast: ``load_components`` logs a component that failed to load and leaves it ``None``.

    Args:
        pipe: The pipeline, after ``load_components``.
        names: The components that must have loaded; every pretrained component of the pipeline when None.

    Raises:
        RuntimeError: If any of them is None.
    """
    wanted = pipe.pretrained_component_names if names is None else names
    missing = [name for name in wanted if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"components failed to load (see the log above): {missing}")
