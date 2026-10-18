"""Profile one Diffu training step to measure Tensor Core (HMMA) utilization.

Applies the torch.profiler method to the *real* model: builds Diffu, runs a few
warmed-up forward+backward steps under bf16 autocast (matching training), then prints
the kernel table plus the share of CUDA time spent in GEMM/conv (Tensor-Core-eligible)
kernels vs memory-bound elementwise/norm/softmax/copy kernels.

The profiling logic itself lives in ``diffu.profiling`` (shared with ``train.py``'s
``--profile-steps`` flag); this script only wires up the real model + synthetic batch.

Reading the output (per the HF "Profiling in PyTorch" posts):
  * kernel names with gemm / tensorop / wmma / s16816 / cutlass  -> Tensor Core math.
  * vectorized_elementwise_kernel / *Norm* / softmax / Memcpy     -> NOT Tensor Core.
  * Self CPU >> Self CUDA  -> overhead-bound (GPU idle, launch-bound).

Run (pin to the GPU with the most free memory; training shares the box):
    CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/profile_tensorcore.py
"""

from __future__ import annotations

import torch

from diffu.config import Config
from diffu.model.diffu import Diffu
from diffu.profiling import profile_callable, read_dcgm


def main() -> None:
    dev = "cuda"
    cfg = Config()
    cfg.aux.diacritic_focal_flow = True  # match the live run (--ink-focal)

    print("building Diffu (loads Qwen VAE + DINOv3 + SD3 backbone)...", flush=True)
    model = Diffu(cfg).to(dev)
    model.train()

    b, h, w = 2, cfg.data.line_height, 512  # small batch: shares a GPU with live training

    def make_batch() -> tuple[torch.Tensor, list[str], torch.Tensor]:
        images = torch.rand(b, 3, h, w, device=dev) * 2 - 1
        texts = ["Rapportbok for aaret 1885"] * b
        style = torch.randn(b, 3, 224, 224, device=dev)
        return images, texts, style

    def step() -> None:
        images, texts, style = make_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(images, texts, style, cond_dropout=0.1)
        loss.backward()
        model.zero_grad(set_to_none=True)

    report = profile_callable(step, warmup=3, active=3)
    print()
    print(report.render(), flush=True)

    dcgm = read_dcgm()
    if dcgm is not None:
        print()
        print(f"==== DCGM (GPU 0): {dcgm} ====", flush=True)


if __name__ == "__main__":
    main()
