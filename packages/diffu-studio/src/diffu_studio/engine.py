"""The stateful engine: resolve the device once, load each model once, hold the optional recogniser.

The UI and the API share one :class:`StudioEngine` for the process lifetime. Models are cached by weights
path so clicking a checkpoint twice doesn't reload it; the cache is capped (default 2, enough for the
single-line A/B compare) and evicts oldest-first with a CUDA cache flush, so switching through many
checkpoints never grows VRAM without bound and never OOMs the co-tenant training run.
"""

from __future__ import annotations

from diffu_studio.checkpoints import ArchConfig, LoadedModel, load_model, resolve_weights
from diffu_studio.device import Device, resolve_device
from diffu_studio.line import Recognizer, load_recognizer


class StudioEngine:
    """Owns the device + a small model cache + the read-back recogniser for the app's lifetime."""

    def __init__(
        self,
        *,
        device: Device | None = None,
        ckpt_root: str = "checkpoints",
        arch: ArchConfig | None = None,
        recognizer_name: str | None = None,
        max_cached_models: int = 2,
    ) -> None:
        self.device = device or resolve_device()
        self.ckpt_root = ckpt_root
        self.arch = arch or ArchConfig()
        self.max_cached_models = max_cached_models
        # keyed by (weights, lora adapter, lora scale) so switching adapter — or base — reloads, not stale
        self._models: dict[tuple[str, str, float], LoadedModel] = {}
        self._recognizer: Recognizer | None = (
            load_recognizer(recognizer_name, self.device.torch_device) if recognizer_name else None
        )

    @property
    def recognizer(self) -> Recognizer | None:
        return self._recognizer

    def model(self, path: str, *, lora: str | None = None, lora_scale: float = 1.0) -> LoadedModel:
        """Return the model at ``path`` (weights file or ``step_*`` dir), loading + caching it on first use.

        An optional Stage-3 ``lora`` adapter (dir/Hub id, applied at ``lora_scale``) is part of the cache
        key, so switching adapter — or scale — reloads instead of returning the base model.
        Raises :class:`~diffu_studio.checkpoints.CheckpointMismatch` if the checkpoint doesn't fit the code.
        """
        key = (resolve_weights(path), lora or "", lora_scale if lora else 1.0)
        cached = self._models.get(key)
        if cached is not None:
            return cached
        loaded = load_model(path, self.arch, self.device, lora=lora, lora_scale=lora_scale)
        # ZeroGPU: opt-in AoT compile of the backbone (aokit). OFF by default (set DIFFU_STUDIO_AOT=1) until
        # the compiled graph's persistence across ZeroGPU's per-request forks is validated — the compile runs
        # in a @spaces.GPU fork whose install would not survive to the parent. Fallback-safe regardless.
        import os

        if os.environ.get("DIFFU_STUDIO_AOT") == "1":
            from diffu_studio.optimization import aot_compile_backbone

            aot_compile_backbone(loaded)
        self._models[key] = loaded
        self._evict_overflow()
        return loaded

    def _evict_overflow(self) -> None:
        while len(self._models) > self.max_cached_models:
            oldest = next(iter(self._models))
            del self._models[oldest]
            _free_cuda()


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — freeing cache is best-effort; never fail an eviction on it
        pass
