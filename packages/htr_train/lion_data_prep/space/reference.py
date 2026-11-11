"""The Space's reference timings (reference.json): measure_reference.py writes them and app.py shows them beside its
live race."""

from __future__ import annotations

from pydantic import BaseModel


class RegionTimes(BaseModel):
    """One region's median seconds, each from the encoded images to the reading: TrOCR on its lines one after the
    other and as one batch, and the reader in vLLM alone and with its DSpark draft."""

    lines: int
    tokens: int
    trocr_single: float
    trocr_batch: float
    vllm: float
    vllm_dspark: float


class Reference(BaseModel):
    """The timings of every example region, with the GPU and the day they were measured on."""

    gpu: str
    measured: str
    runs: int
    regions: dict[str, RegionTimes]
