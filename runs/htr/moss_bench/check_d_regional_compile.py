"""Item D on the GPU: one joint block compiled as regional compile compiles it (``JointLayer.compile`` with SGLang's
compile arguments): it compiles, it compiles again for a page of another size (torch.compile turns the shapes dynamic
on the second size, where SGLang's bit-exact gates broke its shape solver before the traced path), how far it is from
the eager block, and how fast.

Run on GPU 2, quiet, in the SGLang environment (the first compile takes minutes under max-autotune):

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=packages/sglang_moss/src:runs/htr/moss_bench \
        /mnt/scratch/gabriel/envs/sglang_moss/bin/python runs/htr/moss_bench/check_d_regional_compile.py

Inductor computes the fused elementwise chains in fp32 without bf16 rounding between them, so the compiled block is
close to the eager one, not equal. The whole page under regional compile: sglang_bench.py with
{"enable_torch_compile": true, "regional_compile": true} and SGLANG_EXTERNAL_MODEL_PACKAGE=moss_bench_ext.
"""

import os
import time

import torch
from gpu_block import block, inputs, per_page, timed

from diffu_sglang_moss.dit import JointLayer

MODE = os.environ.get("SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")  # SGLang's default


def main() -> None:
    print(torch.cuda.get_device_name(), torch.__version__, MODE)
    layer = block()
    eager = JointLayer(layer, 0)
    compiled = JointLayer(layer, 0)
    compiled.compile(fullgraph=False, dynamic=None, mode=MODE)  # SGLang's build_torch_compile_kwargs
    with torch.no_grad():
        for tokens in (15700, 14100):  # two page sizes: the second compile is the dynamic one
            x, rope, modulation = inputs(tokens=tokens)
            began = time.time()
            out = compiled(x, rope, modulation, frozenset())
            torch.cuda.synchronize()
            print(f"S={tokens}: first call (compiles) {time.time() - began:.1f} s")
            reference = eager(x, rope, modulation, frozenset())
            error = (out.float() - reference.float()).abs()
            print(f"  vs eager: max abs {error.max():.3e}, rel-RMS {error.norm() / reference.float().norm():.3e}")
            for name, module in (("eager", eager), ("compiled", compiled)):
                ms, _ = timed(lambda module=module, x=x, rope=rope, mod=modulation: module(x, rope, mod, frozenset()))
                print(f"  {name:9s} {ms:8.2f} ms a block   {per_page(ms):6.1f} s a page")


if __name__ == "__main__":
    main()
