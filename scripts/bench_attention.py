"""Benchmark diffusers attention backends on the real Diffu backbone — pick the fastest that is correct.

Times each backend for INFERENCE (one denoising step, no grad) and TRAINING (forward + backward), in bf16 on
the GPU, at the real backbone size and real line shapes, and checks its output against torch SDPA (the
default). A backend that is missing, unsupported on this GPU, or has no backward kernel (e.g. Sage) is
reported as such instead of crashing the run. The verdict names the fastest backend whose output matches.

    CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/bench_attention.py
    ... --backbone flux2              # the FLUX.2-style backbone (config.flux2_preset) instead of SD3
    ... --canvas                      # --style-canvas shapes: a reference line next to the target
    ... --compile                     # time the regionally-compiled blocks (what --compile trains with)
    ... --backends native,_native_cudnn,sage_blackwell_hub

Comparing ``--backbone sd3`` and ``--backbone flux2`` runs (same ``--canvas`` / ``--compile``) also compares the
two block designs at equal size. With ``--canvas`` the flux2 inference step reuses the cached reference
K/V, as sampling does after its first step.

Use the winner with ``diffu-train/diffu-generate --attention-backend <name>`` (``native`` = the default).
Hub kernels (``*_hub``) download on first use and may need ``DIFFUSERS_TRUST_REMOTE_KERNELS=true``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch
from pydantic import BaseModel

from diffu.config import BackboneConfig, flux2_preset
from diffu.model.backbone import Backbone
from diffu.model.backbone_flux2 import Flux2Backbone

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
LATENT_C = 16  # Qwen-Image VAE latent channels (the velocity the backbone predicts)


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


def _config(args: argparse.Namespace) -> BackboneConfig:
    if args.tiny:
        cfg = BackboneConfig(dim=64, num_layers=2, heads=2, context_dim=64)
        if args.backbone == "flux2":
            cfg = cfg.model_copy(update={"arch": "flux2", "num_layers": 1, "num_single_layers": 1})
        return cfg
    cfg = flux2_preset(BackboneConfig()) if args.backbone == "flux2" else BackboneConfig()
    return cfg.model_copy(update={"num_layers": args.layers}) if args.layers else cfg


def _build(args: argparse.Namespace, device: torch.device) -> Backbone | Flux2Backbone:
    """The backbone at the benchmark size, with identical (seeded) weights on every call."""
    cfg = _config(args)
    torch.manual_seed(0)
    if args.backbone == "flux2":  # the canvas reference is separate tokens: no extra input channels
        model = Flux2Backbone(cfg, LATENT_C, cfg.context_dim, cfg.context_dim, LATENT_C)
    else:  # sd3 canvas: the reference shares the image grid + rides in LATENT_C conditioning channels
        in_channels = 2 * LATENT_C if args.canvas else LATENT_C
        model = Backbone(cfg, in_channels, cfg.context_dim, cfg.context_dim, LATENT_C)
    return model.to(device)


def _runners(
    args: argparse.Namespace, model: Backbone | Flux2Backbone, device: torch.device
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    """(one inference step, one training forward) on a batch at the real line geometry."""
    ctx_dim = _config(args).context_dim
    h, w = args.line_height // 8, args.width // 8
    ref_w = w if args.canvas else 0  # a reference as wide as the target
    gen = torch.Generator(device="cpu").manual_seed(0)
    flux_ref = args.canvas and isinstance(model, Flux2Backbone)
    in_channels = LATENT_C if flux_ref or not args.canvas else 2 * LATENT_C
    grid_w = w if flux_ref else w + ref_w
    latent = torch.randn(args.batch, in_channels, h, grid_w, generator=gen).to(device)
    ref = torch.randn(args.batch, LATENT_C, h, ref_w, generator=gen).to(device) if flux_ref else None
    n_content = (ref_w + w) // 2  # one glyph token per patch column of the whole canvas
    context = torch.randn(args.batch, n_content, ctx_dim, generator=gen).to(device)
    pooled = torch.randn(args.batch, ctx_dim, generator=gen).to(device)
    t = torch.full((args.batch,), 500.0, device=device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    autocast = lambda: torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda")  # noqa: E731

    if not flux_ref:

        def plain() -> torch.Tensor:
            with autocast():
                return model(latent, t, context, pooled, n_content=n_content)

        return plain, plain
    assert isinstance(model, Flux2Backbone) and ref is not None
    with torch.no_grad(), autocast():
        cache = model.forward_with_ref(latent, t, context, pooled, ref, n_content=n_content)[1]

    def cached_step() -> torch.Tensor:  # every sampling step after the first
        with autocast():
            return model.forward_cached(latent, t, context, pooled, cache, ref_w // 2, n_content=n_content)

    def train_forward() -> torch.Tensor:  # training always recomputes the reference
        with autocast():
            return model.forward_with_ref(latent, t, context, pooled, ref, n_content=n_content)[0]

    return cached_step, train_forward


def _bench(
    args: argparse.Namespace, backend: str, reference: torch.Tensor | None, device: torch.device
) -> Result:
    model = _build(args, device)
    try:
        if backend != "native":
            model.set_attention_backend(backend)
        if args.compile:
            model.compile_blocks()
    except Exception as exc:  # noqa: BLE001 - missing package / kernel / unsupported GPU: report, keep going
        return Result(backend=backend, note=f"unavailable: {type(exc).__name__}: {str(exc)[:80]}")

    result = Result(backend=backend)
    try:
        model.eval()
        infer, train_forward = _runners(args, model, device)
        with torch.no_grad():
            out = infer().float()
            result.infer_ms = _time(infer, args.warmup, args.iters, device)
        if reference is not None:
            result.max_diff = (out - reference).abs().max().item()
    except Exception as exc:  # noqa: BLE001
        return Result(backend=backend, note=f"inference failed: {type(exc).__name__}: {str(exc)[:80]}")

    model.train()

    def train_step() -> None:
        train_forward().float().square().mean().backward()
        model.zero_grad(set_to_none=True)

    try:
        result.train_ms = _time(train_step, args.warmup, args.iters, device)
    except Exception as exc:  # noqa: BLE001 - e.g. Sage has no backward kernel
        result.note = f"no training: {type(exc).__name__}: {str(exc)[:60]}"
    return result


def _reference_output(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """The default-SDPA inference output on the benchmark batch (the correctness reference)."""
    model = _build(args, device).eval()
    infer, _ = _runners(args, model, device)
    with torch.no_grad():
        return infer().float()


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark diffusers attention backends on the Diffu backbone.")
    ap.add_argument("--backends", default=",".join(DEFAULT_BACKENDS), help="comma-separated backend names")
    ap.add_argument("--backbone", choices=("sd3", "flux2"), default="sd3")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024, help="target line width in px")
    ap.add_argument("--line-height", type=int, default=128)
    ap.add_argument(
        "--canvas", action="store_true", help="--style-canvas shapes (a reference next to the target)"
    )
    ap.add_argument("--compile", action="store_true", help="time the regionally-compiled blocks")
    ap.add_argument("--layers", type=int, default=None, help="override the (dual-stream) depth")
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
        cap = "".join(map(str, torch.cuda.get_device_capability(device)))
        print(f"GPU: {torch.cuda.get_device_name(device)} (sm_{cap})")
    else:
        print("no CUDA device: timings are NOT representative (plumbing check only)")
    params = sum(p.numel() for p in _build(args, torch.device("meta")).parameters()) / 1e6
    print(
        f"torch {torch.__version__}  backbone {args.backbone} ({params:.0f}M)  batch {args.batch}  "
        f"width {args.width}px  canvas {args.canvas}  compile {args.compile}\n"
    )

    backends = [b.strip() for b in args.backends.split(",") if b.strip() and b.strip() != "native"]
    baseline = _bench(args, "native", None, device)  # the reference runs first
    reference = _reference_output(args, device) if baseline.infer_ms is not None else None
    results = [baseline] + [_bench(args, b, reference, device) for b in backends]

    def fmt(value: float | None, digits: int = 2) -> str:
        return f"{value:.{digits}f}" if value is not None else "-"

    print(f"{'backend':<22}{'infer ms':>10}{'train ms':>10}{'max diff':>10}  note")
    for r in results:
        print(f"{r.backend:<22}{fmt(r.infer_ms):>10}{fmt(r.train_ms):>10}{fmt(r.max_diff, 4):>10}  {r.note}")

    ok = [r for r in results if r.max_diff is None or r.max_diff <= args.tolerance]
    for phase, key in (("inference", "infer_ms"), ("training", "train_ms")):
        timed = [r for r in ok if getattr(r, key) is not None]
        if timed:
            best = min(timed, key=lambda r: getattr(r, key))
            base = getattr(baseline, key)
            speedup = f" ({base / getattr(best, key):.2f}x vs native)" if base else ""
            print(f"\nfastest correct for {phase}: {best.backend}{speedup}")


if __name__ == "__main__":
    main()
