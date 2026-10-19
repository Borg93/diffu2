"""Benchmark diffusers attention backends on the real Diffu backbone — pick the fastest that is correct.

Times each backend for INFERENCE (forward, no grad) and TRAINING (forward + backward), in bf16 on the GPU,
at the real backbone size and real line shapes, and checks its output against torch SDPA (the default).
A backend that is missing, unsupported on this GPU, or has no backward kernel (e.g. Sage) is reported
as such instead of crashing the run. The verdict names the fastest backend whose output matches.

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/bench_attention.py
    ... --canvas                      # the --style-canvas shapes: [ref | target] canvas, 2x input channels
    ... --compile                     # time the regionally-compiled blocks (what --compile trains with)
    ... --backends native,_native_cudnn,sage_blackwell_hub

Use the winner with ``diffu-train/diffu-generate --attention-backend <name>`` (``native`` = the default).
Hub kernels (``*_hub``) download on first use and may need ``DIFFUSERS_TRUST_REMOTE_KERNELS=true``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch
from pydantic import BaseModel

from diffu.config import BackboneConfig
from diffu.model.backbone import Backbone

DEFAULT_BACKENDS = (
    "native",
    "_native_flash",
    "_native_cudnn",
    "_native_efficient",
    "flash",
    "flash_hub",
    "_flash_3_hub",
    "flash_4_hub",
    "sage",
    "sage_hub",
    "sage_blackwell_hub",
)


class Result(BaseModel):
    backend: str
    infer_ms: float | None = None
    train_ms: float | None = None
    max_diff: float | None = None
    note: str = ""


def _time(fn: Callable[[], object], warmup: int, iters: int, device: torch.device) -> float:
    """Mean wall time per call in ms, synchronised around the timed region."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters


LATENT_C = 16  # Qwen-Image VAE latent channels (the velocity the backbone predicts)


def _build(args: argparse.Namespace, device: torch.device) -> tuple[Backbone, BackboneConfig, int]:
    """The backbone at the benchmark size, with identical (seeded) weights on every call."""
    cfg = BackboneConfig(num_layers=args.layers) if args.layers else BackboneConfig()
    if args.tiny:
        cfg = BackboneConfig(dim=64, num_layers=2, heads=2, context_dim=64)
    in_channels = 2 * LATENT_C if args.canvas else LATENT_C
    torch.manual_seed(0)
    model = Backbone(cfg, in_channels=in_channels, out_channels=LATENT_C)
    return model.to(device), cfg, in_channels


def _inputs(args: argparse.Namespace, in_channels: int, context_dim: int, device: torch.device) -> tuple:
    """A batch at the real line geometry: latent [B, C, H/8, W/8] + one glyph token per patch column."""
    h, w = args.line_height // 8, args.width // 8
    if args.canvas:
        w *= 2  # [reference | target] on one canvas
    gen = torch.Generator(device="cpu").manual_seed(0)
    latent = torch.randn(args.batch, in_channels, h, w, generator=gen)
    context = torch.randn(args.batch, w // 2, context_dim, generator=gen)  # one token per patch column
    pooled = torch.randn(args.batch, context_dim, generator=gen)
    timestep = torch.full((args.batch,), 500.0)
    return tuple(t.to(device) for t in (latent, timestep, context, pooled))


def _bench(
    args: argparse.Namespace, backend: str, reference: torch.Tensor | None, device: torch.device
) -> Result:
    model, cfg, in_channels = _build(args, device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        if backend != "native":
            model.set_attention_backend(backend)
        if args.compile:
            model.compile_blocks()
    except Exception as exc:  # noqa: BLE001 - missing package / kernel / unsupported GPU: report, keep going
        return Result(backend=backend, note=f"unavailable: {type(exc).__name__}: {str(exc)[:80]}")

    latent, timestep, context, pooled = _inputs(args, in_channels, cfg.context_dim, device)

    def forward() -> torch.Tensor:
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            return model(latent, timestep, context, pooled)

    result = Result(backend=backend)
    try:
        model.eval()
        with torch.no_grad():
            out = forward().float()
            result.infer_ms = _time(forward, args.warmup, args.iters, device)
        if reference is not None:
            result.max_diff = (out - reference).abs().max().item()
    except Exception as exc:  # noqa: BLE001
        return Result(backend=backend, note=f"inference failed: {type(exc).__name__}: {str(exc)[:80]}")

    model.train()

    def train_step() -> None:
        forward().float().square().mean().backward()
        model.zero_grad(set_to_none=True)

    try:
        result.train_ms = _time(train_step, args.warmup, args.iters, device)
    except Exception as exc:  # noqa: BLE001 - e.g. Sage has no backward kernel
        result.note = f"no training: {type(exc).__name__}: {str(exc)[:60]}"
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark diffusers attention backends on the Diffu backbone.")
    ap.add_argument("--backends", default=",".join(DEFAULT_BACKENDS), help="comma-separated backend names")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024, help="target line width in px")
    ap.add_argument("--line-height", type=int, default=128)
    ap.add_argument(
        "--canvas", action="store_true", help="--style-canvas shapes ([ref | target], 2x channels)"
    )
    ap.add_argument("--compile", action="store_true", help="time the regionally-compiled blocks")
    ap.add_argument("--layers", type=int, default=None, help="override the backbone depth (default: config)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--tolerance", type=float, default=0.05, help="max |out - SDPA| to count as correct (bf16)"
    )
    ap.add_argument("--tiny", action="store_true", help="tiny backbone (CPU plumbing check, not a benchmark)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(device)} (sm_{''.join(map(str, torch.cuda.get_device_capability(device)))})"
        )
    else:
        print("no CUDA device: timings are NOT representative (plumbing check only)")
    print(
        f"torch {torch.__version__}  batch {args.batch}  width {args.width}px  canvas {args.canvas}  compile {args.compile}\n"
    )

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if "native" in backends:  # the reference must run first
        backends.remove("native")
    baseline = _bench(args, "native", None, device)
    reference = None
    if baseline.infer_ms is not None:  # rerun the reference forward once to get its output for the diffs
        reference = _reference_output(args, device)
    results = [baseline] + [_bench(args, b, reference, device) for b in backends]

    print(f"{'backend':<22}{'infer ms':>10}{'train ms':>10}{'max diff':>10}  note")
    for r in results:
        fmt = lambda v, p=2: f"{v:.{p}f}" if v is not None else "-"  # noqa: E731
        print(f"{r.backend:<22}{fmt(r.infer_ms):>10}{fmt(r.train_ms):>10}{fmt(r.max_diff, 4):>10}  {r.note}")

    ok = [r for r in results if r.max_diff is None or r.max_diff <= args.tolerance]
    for phase, key in (("inference", "infer_ms"), ("training", "train_ms")):
        timed = [r for r in ok if getattr(r, key) is not None]
        if timed:
            best = min(timed, key=lambda r: getattr(r, key))
            base = getattr(baseline, key)
            speedup = f" ({base / getattr(best, key):.2f}x vs native)" if base else ""
            print(f"\nfastest correct for {phase}: {best.backend}{speedup}")


def _reference_output(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """The default-SDPA output on the benchmark batch (the correctness reference)."""
    model, cfg, in_channels = _build(args, device)
    model.eval()
    latent, timestep, context, pooled = _inputs(args, in_channels, cfg.context_dim, device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
        return model(latent, timestep, context, pooled).float()


if __name__ == "__main__":
    main()
