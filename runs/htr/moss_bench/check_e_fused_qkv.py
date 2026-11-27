"""Item E on the GPU: Q/K/V as one GEMM (the plugin's ``fused_qkv``): how far its block is from three GEMMs' in bf16,
and whether it is faster, for the GEMMs alone and for the whole block.

Run on GPU 2, quiet, in the SGLang environment:

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=packages/sglang_moss/src:runs/htr/moss_bench \
        /mnt/scratch/gabriel/envs/sglang_moss/bin/python runs/htr/moss_bench/check_e_fused_qkv.py

The one GEMM's queries and keys are copied contiguous (SGLang's fused RMSNorm + RoPE takes no other), which the
"GEMMs alone" line counts.
"""

import torch
from gpu_block import block, inputs, per_page, timed

from diffu_sglang_moss.dit import _qkv, joint_block, qkv_weight


def main() -> None:
    print(torch.cuda.get_device_name(), torch.__version__)
    layer = block()
    x, rope, modulation = inputs()
    y = torch.randn_like(x)
    with torch.no_grad():
        qkv_weight(layer)  # built once; the three weights become its views
        three = joint_block(layer, x, rope, modulation)
        one = joint_block(layer, x, rope, modulation, fused_qkv=True)
        error = (one.float() - three.float()).abs()
        print(f"block, one GEMM vs three: max abs {error.max():.3e}, rel-RMS {error.norm() / three.float().norm():.3e}")
        print("bit for bit:", torch.equal(one, three))
        for name, run in (
            ("GEMMs alone, three", lambda: _qkv(layer, y, False)),
            ("GEMMs alone, one", lambda: _qkv(layer, y, True)),
            ("block, three GEMMs", lambda: joint_block(layer, x, rope, modulation)),
            ("block, one GEMM", lambda: joint_block(layer, x, rope, modulation, fused_qkv=True)),
        ):
            ms, _ = timed(run)
            print(f"{name:22s} {ms:8.3f} ms   {per_page(ms):6.2f} s a page")


if __name__ == "__main__":
    main()
