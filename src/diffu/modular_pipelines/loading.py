"""Loading a modular pipeline's components without letting one that failed slip through."""

from __future__ import annotations

from collections.abc import Iterable

from diffusers.modular_pipelines import ModularPipeline


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
