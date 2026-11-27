"""Item A on the GPU: the plugin's joint block with SGLang's fused RMSNorm + modulation and gated residual on the two
parts of a sample page's tokens is bit for bit the plain path, its fused kernels ran, and how much faster it is.

Run on GPU 2, quiet, in the SGLang environment (never on CPU: the fused kernels' gates check the device):

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=packages/sglang_moss/src:runs/htr/moss_bench \
        /mnt/scratch/gabriel/envs/sglang_moss/bin/python runs/htr/moss_bench/check_a_fused_block.py

Prints, for one block at the real size (gpu_block.py): whether the fused path equals the plain one (torch.equal),
whether each of SGLang's bit-exact gates verified and stayed on, whether residual_gate_add took its CUDA route on
both parts, and the milliseconds of three paths: the new one, the one before it (27 Nov, 9d71f8f: modulation per
token in plain torch, the rest fused), and the plain one.
"""

import torch
from gpu_block import block, inputs, per_page, timed
from sglang.kernels.kda_kernels.residual_gate_add_jit import can_use_residual_gate_add_cuda
from sglang.multimodal_gen.runtime.models.dits import moss_image as sgl

from diffu_sglang_moss.dit import _by_token, _segments, _traced_joint_block, joint_block


def before(layer, x, rope, modulation):
    """The joint block as of 9d71f8f: the modulation per token in plain torch (no fused modulation kernel)."""
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = _by_token(modulation, x.shape[1]).unbind(2)
    dtype = layer.proj["q"].weight.dtype
    y = (layer.norm1(x) * (1 + scale_a) + shift_a).to(dtype)
    q = sgl._norm_rope(layer.q_norm, layer.proj["q"](y), rope, layer.heads)
    k = sgl._norm_rope(layer.k_norm, layer.proj["k"](y), rope, layer.heads)
    x = x + layer.proj["o"](sgl._attention(q, k, layer.proj["v"](y), None, layer.heads, layer.attention)) * gate_a
    y = layer.norm2(x) * (1 + scale_f) + shift_f
    hidden, gate = layer.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    return x + layer.proj["ffn_down"](sgl._situ(hidden, gate)) * gate_f


def main() -> None:
    print(torch.cuda.get_device_name(), torch.__version__)
    layer = block()
    x, rope, modulation = inputs()
    with torch.no_grad():
        fused = joint_block(layer, x, rope, modulation)  # the first call verifies each gate against eager
        plain = _traced_joint_block(layer, x, rope, modulation)
        print("fused == plain (bit for bit):", torch.equal(fused, plain))
        print("fused == before (bit for bit):", torch.equal(fused, before(layer, x, rope, modulation)))
        for gate in (sgl._NORM_MOD_FUSION, sgl._NORM_ROPE_FUSION, sgl._SITU_FUSION, sgl._RMSNORM_FUSION):
            print(f"  gate {gate.name}: verified={gate.verified} disabled={gate.disabled}")
        gate_a = modulation.image[:, 2, None]
        routes = [can_use_residual_gate_add_cuda(part, part, gate_a) for part, _ in _segments(x, modulation)]
        print("  residual_gate_add on its CUDA route, each part:", routes)
        for name, run in (
            ("new (fused, by part)", lambda: joint_block(layer, x, rope, modulation)),
            ("before (9d71f8f)", lambda: before(layer, x, rope, modulation)),
            ("plain", lambda: _traced_joint_block(layer, x, rope, modulation)),
        ):
            ms, _ = timed(run)
            print(f"{name:24s} {ms:8.2f} ms a block   {per_page(ms):6.1f} s a page")


if __name__ == "__main__":
    main()
