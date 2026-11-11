"""MXFP8 inference (OCP Microscaling Formats v1.0, its E4M3 MXFP8): a linear layer's weight held as blocks of 32 FP8
E4M3 values along its input dimension, each block sharing one E8M0 scale (a power of two), its input quantized the
same way at every call, and the product taken by Blackwell's block-scaled tensor cores
(``torch.nn.functional.scaled_mm``, cuBLAS's MXFP8 GEMM), accumulated in fp32 and returned in the layer's own dtype.

A block's scale is the spec's (section 6.3): the largest power of two at most the block's largest magnitude, divided
by E4M3's largest power of two (2^8); each element is divided by it, rounded to nearest even and clamped to E4M3's
±448, so a magnitude between 448 and 512 times the scale saturates. A block whose scale would fall under E8M0's
smallest, 2^-127 (its largest magnitude under 2^-119: zeros, float32 subnormals), takes that smallest; a block holding
an infinity or a NaN takes the NaN scale (0xFF), so every value of it reads back as NaN.

The scales reach ``scaled_mm`` (``SWIZZLE_32_4_4``) as cuBLAS lays out the scales of a block-scaled GEMM (its "1D
Block Scaling Factors Layout"): tiles of 128 rows by 4 blocks, the tiles row-major, each tile 512 bytes holding row
``r`` and block ``c`` at ``(r % 32) * 16 + (r // 32) * 4 + c``; the rows and the blocks padded with zeros to whole
tiles. A weight's scales are laid out once, when the layer is quantized; an input's at every call, by plain torch
ops that ``torch.compile`` fuses with the quantization.

Away from a Blackwell GPU (on the CPU, in tests) a quantized layer emulates the product in fp32 from the same blocks:
the operands it multiplies are exactly the GPU's, only the order of the fp32 sums differs.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import NamedTuple, Self, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import ScalingType, SwizzleType

BLOCK = 32  # elements per scale: MXFP8's block size
E4M3_MAX = 448.0  # E4M3's largest normal
E4M3_EMAX = 8  # the exponent of E4M3's largest power of two, 2^8
E8M0_NAN = 255
FLOAT32_MANTISSA_BITS = 23
FLOAT32_EXPONENT_MASK = 0xFF
FLOAT32_TOP_EXPONENT = 254  # float32's largest biased exponent of a finite number
TILE_ROWS, TILE_BLOCKS = 128, 4  # cuBLAS's tile of block scales
GEMM_ALIGNMENT = 16  # scaled_mm's inner and output widths are multiples of 16
BLACKWELL = (10, 0)  # the compute capability from which cuBLAS multiplies MXFP8 blocks

# The layers MXFP8 computes: the DiT's joint blocks' six matrices (not its embedders, its text refiner, its modulation
# or its output projection), and every matrix of the text encoder's decoder layers but the delta rule's two gates
# (in_proj_a and in_proj_b: 32 outputs each, nothing to gain, and they set the linear attention's decay); its vision
# tower never runs for a prompt.
JOINT_BLOCK_LINEARS = re.compile(r"proj\.(q|k|v|o|ffn_up_gate|ffn_down)")
DIT_LINEARS = re.compile(rf"layers\.\d+\.{JOINT_BLOCK_LINEARS.pattern}")
TEXT_ENCODER_LINEARS = re.compile(
    r"language_model\.layers\.\d+\."
    r"(self_attn\.[qkvo]_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)"
)


class Precision(StrEnum):
    """What a model's chosen linear layers compute in: their weights' own bfloat16, or MXFP8."""

    BF16 = "bf16"
    MXFP8 = "mxfp8"


class MXBlocks(NamedTuple):
    """A tensor in MXFP8: its elements ``[..., K]`` (FP8 E4M3) and the E8M0 scale of each block of 32 along its last
    dimension ``[..., K / 32]``, as the scale's byte (uint8: the biased exponent, 2^(byte - 127); 255 is NaN)."""

    elements: Tensor
    scales: Tensor


def quantize(x: Tensor) -> MXBlocks:
    """``x`` in MXFP8 along its last dimension (a multiple of 32), by the spec's conversion (the module's docstring).

    The scale's exponent is read from the bits of each block's largest magnitude in float32, exactly; the elements are
    multiplied by the scale's inverse, a power of two, so the only rounding is E4M3's.
    """
    blocks = x.float().unflatten(-1, (-1, BLOCK))
    amax = blocks.abs().amax(-1, keepdim=True)
    biased = (amax.view(torch.int32) >> FLOAT32_MANTISSA_BITS) & FLOAT32_EXPONENT_MASK  # floor(log2 amax) + 127
    exponent = (biased - E4M3_EMAX).clamp(0, FLOAT32_TOP_EXPONENT)
    inverse = ((FLOAT32_TOP_EXPONENT - exponent) << FLOAT32_MANTISSA_BITS).view(torch.float32)  # 2^(127 - exponent)
    elements = (blocks * inverse).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).flatten(-2)
    scales = torch.where(amax.isfinite(), exponent, E8M0_NAN).squeeze(-1).to(torch.uint8)
    return MXBlocks(elements, scales)


def dequantize(blocks: MXBlocks) -> Tensor:
    """MXFP8 blocks as the float32 values they stand for: each element times its block's scale (exact in float32)."""
    scales = blocks.scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    return (blocks.elements.float().unflatten(-1, (-1, BLOCK)) * scales).flatten(-2)


def swizzled(scales: Tensor) -> Tensor:
    """Block scales ``[rows, blocks]`` (row ``r``'s scale of its block ``c``) in cuBLAS's tiled layout, flat: what
    ``scaled_mm`` reads as ``SWIZZLE_32_4_4``. The rows are padded to a multiple of 128 and the blocks to a multiple of
    4 with zeros, as cuBLAS requires; any dtype."""
    rows, blocks = scales.shape
    row_tiles, block_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS, (blocks + TILE_BLOCKS - 1) // TILE_BLOCKS
    padded = F.pad(scales, (0, block_tiles * TILE_BLOCKS - blocks, 0, row_tiles * TILE_ROWS - rows))
    # row r = 128 * tile + 32 * group + lane sits in its tile at lane * 16 + group * 4 + block
    tiles = padded.view(row_tiles, TILE_BLOCKS, TILE_ROWS // TILE_BLOCKS, block_tiles, TILE_BLOCKS)
    return tiles.permute(0, 3, 2, 1, 4).reshape(-1)


def unswizzled(flat: Tensor, rows: int, blocks: int) -> Tensor:
    """The block scales ``[rows, blocks]`` that :func:`swizzled` laid out as ``flat``."""
    row_tiles, block_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS, (blocks + TILE_BLOCKS - 1) // TILE_BLOCKS
    tiles = flat.view(row_tiles, block_tiles, TILE_ROWS // TILE_BLOCKS, TILE_BLOCKS, TILE_BLOCKS)
    return tiles.permute(0, 3, 2, 1, 4).reshape(row_tiles * TILE_ROWS, block_tiles * TILE_BLOCKS)[:rows, :blocks]


def swizzled_size(rows: int, blocks: int) -> int:
    """How many scales :func:`swizzled` lays out for ``rows`` rows of ``blocks`` blocks: whole tiles."""
    return -(-rows // TILE_ROWS) * TILE_ROWS * -(-blocks // TILE_BLOCKS) * TILE_BLOCKS


def block_scaled_mm(rows: Tensor, weight: Tensor, weight_scales: Tensor, out_dtype: torch.dtype) -> Tensor:
    """``rows @ weight.T`` by cuBLAS's MXFP8 GEMM: ``rows`` ``[M, K]`` quantized here, ``weight`` ``[N, K]`` (E4M3) and
    its swizzled scales (uint8) quantized before. Plain torch ops and one ``scaled_mm``: a compiled region fuses the
    quantization and keeps the token count ``M`` symbolic."""
    blocks = quantize(rows)
    return F.scaled_mm(
        blocks.elements,
        weight.t(),
        swizzled(blocks.scales).view(torch.float8_e8m0fnu),
        ScalingType.BlockWise1x32,
        weight_scales.view(torch.float8_e8m0fnu),
        ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=out_dtype,
    )


def supports_mxfp8(device: torch.device) -> bool:
    """Whether ``device`` multiplies MXFP8 blocks natively: a CUDA GPU of compute capability 10.0 or later
    (Blackwell's tensor cores, through cuBLAS)."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= BLACKWELL


def fits_mxfp8(in_features: int, out_features: int) -> bool:
    """Whether a linear layer of these widths can compute in MXFP8: whole blocks of inputs, and outputs as cuBLAS
    aligns them."""
    return in_features % BLOCK == 0 and out_features % GEMM_ALIGNMENT == 0


class Linear(nn.Module):
    """A linear layer in MXFP8 for inference: its weight quantized once (:meth:`from_linear`), its input at every call,
    the product accumulated in fp32 and returned in ``out_dtype``, the layer's former dtype. On a CUDA tensor the
    product is :func:`block_scaled_mm`; elsewhere it is emulated in fp32 from the same blocks.

    The weight is held as two buffers of bytes, so a later ``.to(dtype)`` leaves it alone: ``elements`` (E4M3,
    ``[out_features, in_features]``) and ``scales`` (E8M0, laid out by :func:`swizzled`).

    Raises:
        ValueError: From the constructor, if the widths do not fit MXFP8 (:func:`fits_mxfp8`).
    """

    elements: Tensor
    scales: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        out_dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if not fits_mxfp8(in_features, out_features):
            raise ValueError(
                f"an MXFP8 layer takes a multiple of {BLOCK} inputs and of {GEMM_ALIGNMENT} outputs, "
                f"not {in_features} and {out_features}"
            )
        self.in_features, self.out_features, self.out_dtype = in_features, out_features, out_dtype
        self.register_buffer("elements", torch.empty(out_features, in_features, dtype=torch.uint8, device=device))
        scale_count = swizzled_size(out_features, in_features // BLOCK)
        self.register_buffer("scales", torch.empty(scale_count, dtype=torch.uint8, device=device))
        self.register_buffer("bias", torch.empty(out_features, dtype=out_dtype, device=device) if bias else None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Self:
        """``linear`` quantized, on its device, its product returned in its weight's dtype."""
        weight = linear.weight
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            out_dtype=weight.dtype,
            device=weight.device,
        )
        with torch.no_grad():
            blocks = quantize(weight)
            layer.elements.copy_(blocks.elements.view(torch.uint8))
            layer.scales.copy_(swizzled(blocks.scales))
            if layer.bias is not None and linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def weight_blocks(self) -> MXBlocks:
        """The weight as MXFP8 blocks, its scales ``[out_features, in_features / 32]`` in row order."""
        scales = unswizzled(self.scales, self.out_features, self.in_features // BLOCK)
        return MXBlocks(self.elements.view(torch.float8_e4m3fn), scales)

    @override
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        rows = x.reshape(-1, self.in_features).contiguous()
        if rows.is_cuda:
            out = block_scaled_mm(rows, self.elements.view(torch.float8_e4m3fn), self.scales, self.out_dtype)
        else:
            out = (dequantize(quantize(rows)) @ dequantize(self.weight_blocks()).t()).to(self.out_dtype)
        out = out.view(*x.shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias

    @override
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, out_dtype={self.out_dtype}, mxfp8"


def quantize_linears(model: nn.Module, chosen: re.Pattern[str]) -> list[str]:
    """Every ``nn.Linear`` of ``model`` whose name ``chosen`` matches in full replaced, in place, by its MXFP8
    :class:`Linear`. Every chosen layer is checked before any is replaced, so a refusal changes nothing.

    Returns:
        The names of the layers replaced, in module order.

    Raises:
        ValueError: If no layer matches, or a chosen layer does not fit MXFP8 (:func:`fits_mxfp8`).
    """
    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and chosen.fullmatch(name)
    }
    if not layers:
        raise ValueError(f"no linear layer of {type(model).__name__} is named as {chosen.pattern!r}")
    misfits = [name for name, layer in layers.items() if not fits_mxfp8(layer.in_features, layer.out_features)]
    if misfits:
        raise ValueError(
            f"MXFP8 takes layers of a multiple of {BLOCK} inputs and {GEMM_ALIGNMENT} outputs; {len(misfits)} chosen "
            f"layers are not, the first {misfits[0]}"
        )
    for name, layer in layers.items():
        parent, _, child = name.rpartition(".")
        model.get_submodule(parent).register_module(child, Linear.from_linear(layer))
    return list(layers)
