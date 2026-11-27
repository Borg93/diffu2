"""Item G on the GPU: with moss_bench_ext's MOSS_BENCH_FA4_SM120=1, SGLang's "fa" attention backend resolves to
FlashAttention version 4 on this 12.x card; its sm120 kernel against cuDNN SDPA (and torch's default SDPA, which "fa"
was before) on one MOSS-Image attention call, B=1, S=15700, H=24, D=128, bf16, through SGLang's own FlashAttentionImpl.

Run on GPU 2, quiet, in the SGLang environment:

    CUDA_VISIBLE_DEVICES=2 MOSS_BENCH_FA4_SM120=1 PYTHONPATH=packages/sglang_moss/src:runs/htr/moss_bench \
        /mnt/scratch/gabriel/envs/sglang_moss/bin/python runs/htr/moss_bench/check_g_fa4_sm120.py [S]

bench_attn_sm120.py times the same kernels called directly; this one checks the switch's path through SGLang. The
whole page: sglang_bench.py with {"attention_backend": "fa"}, MOSS_BENCH_FA4_SM120=1 and
SGLANG_EXTERNAL_MODEL_PACKAGE=moss_bench_ext.
"""

import math
import os
import sys
from pkgutil import resolve_name

import moss_bench_ext  # noqa: F401  the switch, read from the environment
import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.layers.attention.backends import flash_attn
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum, current_platform
from torch.nn.attention import SDPBackend, sdpa_kernel

S, H, D = int(sys.argv[1]) if len(sys.argv) > 1 else 15700, 24, 128
WARMUP, ITERS = 3, 20


def timed(run):
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        out = run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS, out


def main() -> None:
    assert os.environ.get("MOSS_BENCH_FA4_SM120") == "1", "set MOSS_BENCH_FA4_SM120=1"
    capability = torch.cuda.get_device_capability()
    print(torch.cuda.get_device_name(), capability, torch.__version__, f"S={S} H={H} D={D}")
    resolved = current_platform.get_attn_backend_cls_str(AttentionBackendEnum.FA, D, torch.bfloat16)
    print("'fa' resolves to:", resolved, "| FA version:", flash_attn.fa_ver)
    assert resolved.endswith("FlashAttentionBackend") and flash_attn.fa_ver == 4, "the switch did not take"
    impl = resolve_name(resolved).get_impl_cls()(num_heads=H, head_size=D, causal=False, softmax_scale=1 / math.sqrt(D))
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        reference = F.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q, k, v))).transpose(1, 2)

    def sdpa(backend=None):
        def run():
            if backend is None:  # torch's own pick, as SGLang's torch_sdpa backend leaves it
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                ).transpose(1, 2)
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                ).transpose(1, 2)

        return run

    flops = 4 * S * S * D * H
    for name, run in (
        ("SGLang 'fa' = FA4 sm120", lambda: impl.forward(q, k, v)),
        ("torch cuDNN SDPA", sdpa(SDPBackend.CUDNN_ATTENTION)),
        ("torch SDPA (default pick)", sdpa()),
    ):
        try:
            ms, out = timed(run)
        except Exception as error:  # report and go on
            print(f"{name:28s} failed: {type(error).__name__}: {str(error)[:200]}")
            continue
        out = out[0] if isinstance(out, tuple) else out
        error = (out.float() - reference).norm() / reference.norm()
        print(f"{name:28s} {ms:8.2f} ms {flops / ms / 1e9:7.1f} TFLOP/s  rel-RMS {error:.2e}")


if __name__ == "__main__":
    main()
