"""Standing profiling instrumentation for Diffu (Tensor-Core utilization + DCGM telemetry).

Behind ``diffu-train --profile-steps`` (the trace it writes also feeds the ``profile-model`` skill). The
torch.profiler method (per the HF "Profiling in PyTorch" posts) classifies each CUDA kernel as Tensor-Core-eligible
(GEMM/conv) or memory-bound, then reports the share of GPU time spent doing Tensor-Core math.

Reading the output:
  * kernel names with gemm / tensorop / wmma / s16816 / cutlass  -> Tensor Core math.
  * vectorized_elementwise_kernel / *Norm* / softmax / Memcpy     -> NOT Tensor Core.
  * Self CPU >> Self CUDA  -> overhead-bound (GPU idle, launch-bound).

``torch`` is imported lazily inside the functions that need it, so importing this module is cheap
and safe (it never touches CUDA on import).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from torch.profiler import profile as TorchProfile

TENSOR_CORE_RE = re.compile(
    "gemm|tensorop|wmma|s16816|s161616|cutlass|implicit|dgrad|wgrad|fprop|cublas",
    re.I,
)
"""Kernel-name patterns that indicate Tensor-Core-eligible (GEMM/conv) math."""


class KernelRow(BaseModel):
    """One GPU kernel's self-CUDA time and whether it is Tensor-Core-eligible.

    Attributes:
        name: Profiler kernel key (the demangled kernel name).
        self_cuda_ms: Self GPU time in milliseconds (excludes child kernels).
        tensor_core: True if the name matches :data:`TENSOR_CORE_RE` (GEMM/conv math).
    """

    name: str
    self_cuda_ms: float
    tensor_core: bool


class ProfileReport(BaseModel):
    """Result of profiling a step: the kernel table plus the Tensor-Core breakdown.

    Attributes:
        table_text: The full ``key_averages().table()`` text (top kernels by CUDA time).
        tensor_core_pct: Share of self-CUDA time in Tensor-Core-eligible kernels (0-100).
        total_cuda_ms: Total self-CUDA time across all kernels, in milliseconds.
        top_kernels: Kernels sorted by descending self-CUDA time (for the human summary).
    """

    table_text: str
    tensor_core_pct: float
    total_cuda_ms: float
    top_kernels: list[KernelRow]

    def render(self) -> str:
        """Build the human-readable summary (profiler table + Tensor-Core breakdown).

        Returns:
            A multi-line string ready to print or write to a file.
        """
        lines = [
            "==== PROFILER TABLE (top kernels by CUDA time) ====",
            self.table_text,
            "",
            f"==== GEMM/conv (Tensor-Core-eligible) self CUDA time: "
            f"{self.tensor_core_pct:.1f}% of {self.total_cuda_ms:.1f} ms ====",
            "Top 15 GPU kernels by self CUDA time (TC = Tensor-Core-eligible):",
        ]
        for row in self.top_kernels[:15]:
            tag = "TC " if row.tensor_core else "   "
            lines.append(f"  {tag}{row.self_cuda_ms:8.3f} ms  {row.name[:90]}")
        return "\n".join(lines)


def _build_report(prof: TorchProfile) -> ProfileReport:
    """Assemble a :class:`ProfileReport` from a finished profiler context.

    Args:
        prof: A finished ``torch.profiler.profile`` context with CUDA activity recorded.

    Returns:
        The populated report (table text, Tensor-Core percentage, total time, sorted kernels).
    """
    from torch.autograd import DeviceType

    events = prof.key_averages()
    table_text = events.table(sort_by="device_time_total", row_limit=30)
    # GPU kernels only — the rule torch's own table total uses: the aten op that launched a kernel carries
    # that kernel's time as its own self device time, and user annotations (ProfilerStep#) span them all, so
    # summing every row would count each kernel two or more times.
    rows = [
        (e.key, float(e.self_device_time_total))
        for e in events
        if e.device_type == DeviceType.CUDA and not e.is_user_annotation and e.self_device_time_total > 0
    ]
    total_us = sum(t for _, t in rows)
    kernels = [
        KernelRow(
            name=name,
            self_cuda_ms=t / 1000.0,
            tensor_core=bool(TENSOR_CORE_RE.search(name)),
        )
        for name, t in sorted(rows, key=lambda x: -x[1])
    ]
    pct = (sum(t for name, t in rows if TENSOR_CORE_RE.search(name)) / total_us * 100.0) if total_us else 0.0
    return ProfileReport(
        table_text=table_text,
        tensor_core_pct=pct,
        total_cuda_ms=total_us / 1000.0,
        top_kernels=kernels,
    )


def profile_callable(
    step: Callable[[], None],
    *,
    warmup: int = 2,
    active: int = 3,
    trace_path: str | None = None,
    with_stack: bool = False,
) -> ProfileReport:
    """Profile a step function with torch.profiler (CPU + CUDA) and build a report.

    Runs ``warmup`` un-recorded iterations (to let cuDNN/cuBLAS pick algorithms and warm caches)
    then ``active`` recorded iterations under a ``schedule(wait=0, warmup, active, repeat=1)``.
    Synchronizes CUDA before building the report so all kernel times are flushed.

    Args:
        step: A zero-arg callable that runs exactly one forward+backward step (no optimizer state
            assumptions). It is called ``warmup + active`` times.
        warmup: Number of un-recorded warmup iterations.
        active: Number of recorded iterations to profile.
        trace_path: If set, export the Kineto chrome trace here (for HTA / Perfetto).
        with_stack: Record Python source stacks so ops can be attributed to the line that issued them.
            Writes a stack-grouped CPU-time table next to ``trace_path`` (``.stacks.txt``). Adds
            overhead, so it is off by default and only enabled for a deliberate attribution profile.

    Returns:
        A :class:`ProfileReport` summarizing kernel times and Tensor-Core utilization.
    """
    import torch
    import torch.profiler as prof_mod

    sched = prof_mod.schedule(wait=0, warmup=warmup, active=active, repeat=1)
    with prof_mod.profile(
        activities=[prof_mod.ProfilerActivity.CPU, prof_mod.ProfilerActivity.CUDA],
        schedule=sched,
        with_stack=with_stack,
    ) as prof:
        for _ in range(warmup + active):
            step()
            prof.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)  # Kineto chrome trace for HTA / the profile-model skill
        if with_stack:  # attribute self-CPU time (the cast/launch storm) to the Python line that issued it
            # include_python_functions: torch 2.14 drops Python-function events from key_averages by default.
            averages = prof.key_averages(group_by_stack_n=12, include_python_functions=True)
            table = averages.table(sort_by="self_cpu_time_total", row_limit=40)
            with open(trace_path + ".stacks.txt", "w") as fh:
                fh.write(table)
    return _build_report(prof)


def read_dcgm(fields: tuple[int, ...] = (1002, 1004, 1005)) -> dict[str, float] | None:
    """Read DCGM profiling fields for GPU 0 via ``dcgmi dmon`` (no GPU memory allocated).

    Shells out to ``dcgmi dmon -c 1 -e <fields>`` and parses the SM-activity / Tensor-Core-activity /
    DRAM-activity columns. The default fields are:

      * 1002 — SMACT (SM activity, fraction of time SMs were busy).
      * 1004 — TENSO (Tensor-Core pipe active, fraction of cycles).
      * 1005 — DRAMA (DRAM active, fraction of cycles — memory-bandwidth pressure).

    This reads telemetry counters only; it does NOT launch a CUDA context or allocate device memory,
    so it is safe to call while another job owns the GPUs.

    Args:
        fields: DCGM field ids to request (defaults to SMACT/TENSO/DRAMA).

    Returns:
        A dict mapping ``"SMACT"`` / ``"TENSO"`` / ``"DRAMA"`` (only the parseable ones) to floats,
        or ``None`` if the ``dcgmi`` binary is not installed.
    """
    if shutil.which("dcgmi") is None:
        return None
    field_arg = ",".join(str(f) for f in fields)
    try:
        out = subprocess.run(
            ["dcgmi", "dmon", "-c", "1", "-e", field_arg],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    header: list[str] = []
    result: dict[str, float] = {}
    for line in out.stdout.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] in {"#Entity", "Entity", "ID"} or line.lstrip().startswith("#"):
            header = [t for t in tokens if t not in {"#Entity", "Entity", "ID", "GPU-I", "GPU"}]
            continue
        if tokens[0] in {"GPU", "GPU-I"} and len(tokens) >= 2:
            values = tokens[2:]  # drop "GPU" and the device index
            for name, raw in zip(header, values, strict=False):
                try:
                    result[name] = float(raw)
                except ValueError:
                    continue
            if result:  # first data row (GPU 0) is enough
                break
    return result or None
