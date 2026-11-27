"""Time dense attention kernels on one MOSS-Image DiT call: B=1, S=15700, H=24, D=128, bf16, no mask.

Run on GPU 2 only, on a quiet GPU:  CUDA_VISIBLE_DEVICES=2 python -I bench_attn_sm120.py
Each candidate is tried separately; one that is missing or fails prints why and is skipped.
Error is measured against an fp32 reference (PyTorch's memory-efficient kernel in fp32).
"""

import math
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

S, H, D = int(sys.argv[1]) if len(sys.argv) > 1 else 15700, 24, 128
WARMUP, ITERS = 3, 20
torch.manual_seed(0)
dev = torch.device("cuda")
# [B, S, H, D] like the DiT hands over; SDPA wants [B, H, S, D] views of the same storage
q, k, v = (torch.randn(1, S, H, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
scale = 1.0 / math.sqrt(D)
flops = 4 * S * S * D * H


def sdpa(backend):
    def run():
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)

    return run


def fa4_upstream():
    from flash_attn.cute import flash_attn_func  # flash-attn-4 (SM120 = SM80 mma.sync path)

    def run():
        out = flash_attn_func(q, k, v, softmax_scale=scale, causal=False)
        return out[0] if isinstance(out, tuple) else out

    return run


def fa4_sglang_sm120():
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

    cu = torch.tensor([0, S], dtype=torch.int32, device=dev)

    def run():
        out = flash_attn_varlen_func(
            q[0], k[0], v[0], cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=S, max_seqlen_k=S, softmax_scale=scale, causal=False, ver=4,
        )
        out = out[0] if isinstance(out, tuple) else out
        return out.unsqueeze(0)

    return run


def fp8_sglang_sm120():
    from sglang.kernels.ops.attention.fp8_fa_sm120 import fp8_attention

    def run():
        return fp8_attention(q[0], k[0], v[0], softmax_scale=scale)[0].unsqueeze(0)

    return run


def sage2():
    from sageattention import sageattn

    def run():
        return sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=scale)

    return run


def timed(run):
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(ITERS):
        out = run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS, out


with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    ref = F.scaled_dot_product_attention(
        *(t.float().transpose(1, 2) for t in (q, k, v))
    ).transpose(1, 2)

print(torch.cuda.get_device_name(), torch.__version__, f"S={S} H={H} D={D}")
candidates = {
    "torch flash (FA2 2.5.7)": lambda: sdpa(SDPBackend.FLASH_ATTENTION),
    "torch cuDNN": lambda: sdpa(SDPBackend.CUDNN_ATTENTION),
    "torch mem-efficient": lambda: sdpa(SDPBackend.EFFICIENT_ATTENTION),
    "flash-attn-4 upstream": fa4_upstream,
    "sglang FA4 sm120 (vendored)": fa4_sglang_sm120,
    "sglang fp8_fa_sm120 (lossy)": fp8_sglang_sm120,
    "sageattention 2 (lossy)": sage2,
}
for name, make in candidates.items():
    try:
        ms, out = timed(make())
        err = (out.float() - ref).norm() / ref.norm()
        print(f"{name:30s} {ms:8.2f} ms {flops / ms / 1e9:7.1f} TFLOP/s  rel-RMS {err:.2e}")
    except Exception as exc:  # report and go on
        print(f"{name:30s} skipped: {type(exc).__name__}: {str(exc)[:160]}")
