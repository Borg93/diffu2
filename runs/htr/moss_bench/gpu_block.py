"""One MOSS-Image joint block at the released model's size on the GPU, with a page's tokens, for the check scripts of
this folder (check_*.py); not a test, and never run on CPU.

The block is SGLang's ``_Block`` (dim 3072, feed-forward 8192, 24 heads) with random weights in bf16; its attention is
torch SDPA (SGLang's LocalAttention needs a running server's arguments, and every path checked here calls the same
attention, so it does not change what is compared). The tokens: ``TOKENS`` (15700, B=1), the last ``REFERENCES`` of
them a sample page's (6144, a 1024 x 1536 px page), modulated at timestep 0, the rest at the flow timestep.
"""

import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.models.dits import moss_image as sgl
from torch import Tensor, nn

from diffu_sglang_moss.dit import TokenModulation

DIM, FFN, HEADS, HEAD = 3072, 8192, 24, 128
TOKENS, REFERENCES = 15700, 6144
LAYERS, PASSES, STEPS = 42, 2, 28  # a page: 42 blocks, two passes of guidance, 28 steps


class SDPA(nn.Module):
    """Stands for SGLang's LocalAttention: torch SDPA on ``[B, S, H, D]``."""

    def __init__(self, num_heads: int, head_size: int, **_: object) -> None:
        super().__init__()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def block(seed: int = 0) -> sgl._Block:
    """SGLang's joint block on the GPU in bf16, its weights random (the norms' scales about 1)."""
    sgl.LocalAttention = SDPA
    torch.manual_seed(seed)
    layer = sgl._Block(DIM, FFN, HEADS)
    for parameter in layer.parameters():
        if parameter.ndim == 2:
            nn.init.normal_(parameter, std=parameter.shape[1] ** -0.5)
        else:
            nn.init.normal_(parameter, mean=1.0, std=0.1)
    return layer.to("cuda", torch.bfloat16).eval()


def inputs(tokens: int = TOKENS, references: int = REFERENCES, seed: int = 1):
    """Hidden states ``[1, tokens, dim]``, the rotary coefficients (sin, cos) ``[1, tokens, 128]`` each, and the
    modulation in two parts."""
    generator = torch.Generator("cuda").manual_seed(seed)

    def normal(*shape: int, scale: float = 1.0) -> Tensor:
        return (scale * torch.randn(*shape, device="cuda", generator=generator)).to(torch.bfloat16)

    x = normal(1, tokens, DIM)
    phase = 100 * torch.rand(1, tokens, HEAD // 2, device="cuda", generator=generator)
    sin, cos = phase.sin().to(torch.bfloat16), phase.cos().to(torch.bfloat16)
    rope = (torch.cat((sin, sin), -1).contiguous(), torch.cat((cos, cos), -1).contiguous())
    modulation = TokenModulation(tokens - references, normal(1, 6, DIM, scale=0.1), normal(1, 6, DIM, scale=0.1))
    return x, rope, modulation


def timed(run, iters: int = 10, warmup: int = 3) -> tuple[float, Tensor]:
    """Milliseconds a call of ``run`` (CUDA events, after ``warmup`` calls) and its last output."""
    for _ in range(warmup):
        out = run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, out


def per_page(ms: float) -> float:
    """Seconds a page spends in a block that takes ``ms``."""
    return ms * LAYERS * PASSES * STEPS / 1000
