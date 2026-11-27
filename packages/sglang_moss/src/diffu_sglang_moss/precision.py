"""The precision of the DiT's joint-block projections, chosen for the whole server by ``DIFFU_SGLANG_DIT_PRECISION``.

SGLang 0.5.23 refuses its own quantization for MOSS-Image (``MOSSImageTransformer`` raises on a ``quant_config``),
so the plugin swaps the six projections of every joint block (``layers.N.proj.q``, ``k``, ``v``, ``o``,
``ffn_up_gate``, ``ffn_down``) after the weights are loaded (``DiffuMOSSImageTransformer.post_load_weights``). The
embedders, the text refiner, the time embedding, the modulations, the norms and the final head stay in bfloat16.

- ``bf16`` (the default): nothing is swapped; the DiT is SGLang's as loaded.
- ``mxfp8``: :class:`MXFP8Linear`, OCP MXFP8 (E4M3 elements, one E8M0 scale a block of 32 inputs) for the weight and,
  at every call, for the input; the product by Blackwell's block-scaled tensor cores (``F.scaled_mm``, cuBLAS),
  accumulated in fp32. The conversion is ``diffu_moss.mxfp8``'s, copied here (``quantize``, ``dequantize``,
  ``swizzled``, ``unswizzled``): the plugin's environment has no ``diffu_moss``, and a copy kept identical lets SGLang
  draw on CPU bit for bit what our pipeline draws with ``quantize_linears(transformer, DIT_LINEARS)``.
- ``fp8_rowwise``: :class:`RowwiseFP8Linear`, E4M3 weights with one fp32 scale an output row (its largest magnitude
  over 448) and the input quantized the same way a token at every call; the product by ``torch._scaled_mm`` with
  row-wise scales.

Inputs are quantized by plain torch ops. Under ``torch.compile`` (the server's, or regional compile) Inductor fuses them
into the region; on a GPU outside a compiled region they run as their own small compiled function (built on the first
call, dynamic in the token count), since a dozen separate kernels over the activations would cost more than the faster
GEMM saves. Away from a GPU (on the CPU, in tests) MXFP8's product is emulated in fp32 from the same blocks, as
``diffu_moss.mxfp8`` does, and the row-wise product is ``torch._scaled_mm``'s CPU kernel.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Callable
from enum import StrEnum
from typing import NamedTuple, Self, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import ScalingType, SwizzleType

PRECISION_ENV = "DIFFU_SGLANG_DIT_PRECISION"
JOINT_BLOCK_LINEARS = re.compile(r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)")  # diffu_moss's DIT_LINEARS

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
ADA = (8, 9)  # the compute capability from which torch multiplies FP8 with row-wise scales
ROW_SCALE_FLOOR = 1e-12  # a row of zeros takes this scale, not 0


class DitPrecision(StrEnum):
    """What the DiT's joint-block projections compute in."""

    BF16 = "bf16"
    MXFP8 = "mxfp8"
    FP8_ROWWISE = "fp8_rowwise"


def dit_precision() -> DitPrecision:
    """The precision ``DIFFU_SGLANG_DIT_PRECISION`` names (``bf16`` when unset or empty).

    Raises:
        ValueError: If it names another.
    """
    value = os.environ.get(PRECISION_ENV, "").strip().lower() or DitPrecision.BF16.value
    try:
        return DitPrecision(value)
    except ValueError:
        choices = ", ".join(precision.value for precision in DitPrecision)
        raise ValueError(f"{PRECISION_ENV}={value!r}: the DiT's projections compute in one of {choices}") from None


class MXBlocks(NamedTuple):
    """A tensor in MXFP8: its elements ``[..., K]`` (FP8 E4M3) and the E8M0 scale of each block of 32 along its last
    dimension ``[..., K / 32]``, as the scale's byte (uint8: 2^(byte - 127); 255 is NaN)."""

    elements: Tensor
    scales: Tensor


def quantize(x: Tensor) -> MXBlocks:
    """``x`` in MXFP8 along its last dimension (a multiple of 32), by the OCP spec's conversion: a block's scale is the
    largest power of two at most its largest magnitude over 2^8, read exactly from the float32 bits; each element is
    multiplied by the scale's inverse, rounded to E4M3 and clamped to ±448. ``diffu_moss.mxfp8.quantize``."""
    blocks = x.float().unflatten(-1, (-1, BLOCK))
    amax = blocks.abs().amax(-1, keepdim=True)
    biased = (amax.view(torch.int32) >> FLOAT32_MANTISSA_BITS) & FLOAT32_EXPONENT_MASK  # floor(log2 amax) + 127
    exponent = (biased - E4M3_EMAX).clamp(0, FLOAT32_TOP_EXPONENT)
    inverse = ((FLOAT32_TOP_EXPONENT - exponent) << FLOAT32_MANTISSA_BITS).view(torch.float32)  # 2^(127 - exponent)
    elements = (blocks * inverse).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).flatten(-2)
    scales = torch.where(amax.isfinite(), exponent, E8M0_NAN).squeeze(-1).to(torch.uint8)
    return MXBlocks(elements, scales)


def dequantize(blocks: MXBlocks) -> Tensor:
    """MXFP8 blocks as the float32 values they stand for (exact)."""
    scales = blocks.scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    return (blocks.elements.float().unflatten(-1, (-1, BLOCK)) * scales).flatten(-2)


def swizzled(scales: Tensor) -> Tensor:
    """Block scales ``[rows, blocks]`` in cuBLAS's tiled layout, flat (``SWIZZLE_32_4_4``): tiles of 128 rows by 4
    blocks, row-major, row ``r`` and block ``c`` of a tile at ``(r % 32) * 16 + (r // 32) * 4 + c``; padded with
    zeros to whole tiles."""
    rows, blocks = scales.shape
    row_tiles, block_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS, (blocks + TILE_BLOCKS - 1) // TILE_BLOCKS
    padded = F.pad(scales, (0, block_tiles * TILE_BLOCKS - blocks, 0, row_tiles * TILE_ROWS - rows))
    tiles = padded.view(row_tiles, TILE_BLOCKS, TILE_ROWS // TILE_BLOCKS, block_tiles, TILE_BLOCKS)
    return tiles.permute(0, 3, 2, 1, 4).reshape(-1)


def unswizzled(flat: Tensor, rows: int, blocks: int) -> Tensor:
    """The block scales ``[rows, blocks]`` that :func:`swizzled` laid out as ``flat``."""
    row_tiles, block_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS, (blocks + TILE_BLOCKS - 1) // TILE_BLOCKS
    tiles = flat.view(row_tiles, block_tiles, TILE_ROWS // TILE_BLOCKS, TILE_BLOCKS, TILE_BLOCKS)
    return tiles.permute(0, 3, 2, 1, 4).reshape(row_tiles * TILE_ROWS, block_tiles * TILE_BLOCKS)[:rows, :blocks]


def swizzled_size(rows: int, blocks: int) -> int:
    """How many scales :func:`swizzled` lays out for ``rows`` rows of ``blocks`` blocks."""
    return -(-rows // TILE_ROWS) * TILE_ROWS * -(-blocks // TILE_BLOCKS) * TILE_BLOCKS


def mx_rows(rows: Tensor) -> tuple[Tensor, Tensor]:
    """Rows ``[M, K]`` as ``scaled_mm`` takes an MXFP8 operand: E4M3 elements and their swizzled E8M0 scales."""
    blocks = quantize(rows)
    return blocks.elements, swizzled(blocks.scales).view(torch.float8_e8m0fnu)


def fp8_rows(rows: Tensor) -> tuple[Tensor, Tensor]:
    """Rows ``[M, K]`` in E4M3 with one fp32 scale a row ``[M, 1]``: the row's largest magnitude over 448."""
    wide = rows.float()
    scales = (wide.abs().amax(-1, keepdim=True) / E4M3_MAX).clamp(min=ROW_SCALE_FLOOR)
    return (wide / scales).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scales


@functools.cache
def _compiled(quantizer: Callable[[Tensor], tuple[Tensor, Tensor]]) -> Callable[[Tensor], tuple[Tensor, Tensor]]:
    return torch.compile(quantizer, dynamic=True)


def quantized_input(quantizer: Callable[[Tensor], tuple[Tensor, Tensor]], rows: Tensor) -> tuple[Tensor, Tensor]:
    """``quantizer(rows)``: traced into the region under ``torch.compile``, as its own compiled function on a GPU
    outside one, its plain ops elsewhere."""
    if rows.is_cuda and not torch.compiler.is_compiling():
        return _compiled(quantizer)(rows)
    return quantizer(rows)


def fits_mxfp8(in_features: int, out_features: int) -> bool:
    """Whether a linear layer of these widths can compute in MXFP8: whole blocks of inputs, aligned outputs."""
    return in_features % BLOCK == 0 and out_features % GEMM_ALIGNMENT == 0


def fits_fp8_rowwise(in_features: int, out_features: int) -> bool:
    """Whether a linear layer of these widths can compute in row-wise FP8: ``_scaled_mm``'s alignment."""
    return in_features % GEMM_ALIGNMENT == 0 and out_features % GEMM_ALIGNMENT == 0


class QuantizedLinear(nn.Module):
    """A bias-free linear layer whose weight is held quantized (as bytes, so a later ``.to(dtype)`` leaves it alone),
    its product returned in ``out_dtype``, the replaced layer's dtype.

    Raises:
        ValueError: From the constructor, if the replaced layer has a bias (no joint-block projection has).
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool, out_dtype: torch.dtype) -> None:
        super().__init__()
        if bias:
            raise ValueError("a quantized projection takes no bias: MOSS-Image's joint blocks have none")
        self.in_features, self.out_features, self.out_dtype = in_features, out_features, out_dtype

    def _rows(self, x: Tensor) -> Tensor:
        return x.reshape(-1, self.in_features).contiguous()

    def _shaped(self, out: Tensor, x: Tensor) -> Tensor:
        return out.view(*x.shape[:-1], self.out_features)


def _work_device(weight: Tensor) -> torch.device:
    """Where a weight is quantized: its own device, or the GPU when it was loaded to the host (an offloaded DiT)."""
    if weight.device.type == "cpu" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return weight.device


class MXFP8Linear(QuantizedLinear):
    """A joint-block projection in MXFP8 (``diffu_moss.mxfp8.Linear``'s computation): ``elements`` (E4M3 bytes,
    ``[out_features, in_features]``) and ``scales`` (E8M0 bytes, :func:`swizzled`).

    Raises:
        ValueError: From the constructor, if the widths do not fit MXFP8 (:func:`fits_mxfp8`).
    """

    elements: Tensor
    scales: Tensor

    def __init__(
        self, in_features: int, out_features: int, *, bias: bool, out_dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, out_dtype=out_dtype)
        if not fits_mxfp8(in_features, out_features):
            raise ValueError(
                f"an MXFP8 layer takes a multiple of {BLOCK} inputs and of {GEMM_ALIGNMENT} outputs, "
                f"not {in_features} and {out_features}"
            )
        self.register_buffer("elements", torch.empty(out_features, in_features, dtype=torch.uint8, device=device))
        scale_count = swizzled_size(out_features, in_features // BLOCK)
        self.register_buffer("scales", torch.empty(scale_count, dtype=torch.uint8, device=device))

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Self:
        """``linear`` quantized, kept on its device, its product returned in its weight's dtype."""
        weight = linear.weight
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            out_dtype=weight.dtype,
            device=weight.device,
        )
        with torch.no_grad():
            blocks = quantize(weight.detach().to(_work_device(weight)))
            layer.elements.copy_(blocks.elements.view(torch.uint8))
            layer.scales.copy_(swizzled(blocks.scales))
        return layer

    def weight_blocks(self) -> MXBlocks:
        """The weight as MXFP8 blocks, its scales ``[out_features, in_features / 32]`` in row order."""
        scales = unswizzled(self.scales, self.out_features, self.in_features // BLOCK)
        return MXBlocks(self.elements.view(torch.float8_e4m3fn), scales)

    @override
    def forward(self, x: Tensor) -> Tensor:
        rows = self._rows(x)
        if not rows.is_cuda:
            out = (dequantize(quantize(rows)) @ dequantize(self.weight_blocks()).t()).to(self.out_dtype)
            return self._shaped(out, x)
        elements, scales = quantized_input(mx_rows, rows)
        out = F.scaled_mm(
            elements,
            self.elements.view(torch.float8_e4m3fn).t(),
            scales,
            ScalingType.BlockWise1x32,
            self.scales.view(torch.float8_e8m0fnu),
            ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            output_dtype=self.out_dtype,
        )
        return self._shaped(out, x)

    @override
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, out_dtype={self.out_dtype}, mxfp8"


class RowwiseFP8Linear(QuantizedLinear):
    """A joint-block projection in FP8 with row-wise scales: ``elements`` (E4M3 bytes, ``[out_features,
    in_features]``) and ``scales`` (each output row's fp32 scale ``[1, out_features]``, held as its int32 bits).

    Raises:
        ValueError: From the constructor, if the widths do not fit (:func:`fits_fp8_rowwise`).
    """

    elements: Tensor
    scales: Tensor

    def __init__(
        self, in_features: int, out_features: int, *, bias: bool, out_dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, out_dtype=out_dtype)
        if not fits_fp8_rowwise(in_features, out_features):
            raise ValueError(
                f"a row-wise FP8 layer takes multiples of {GEMM_ALIGNMENT} inputs and outputs, "
                f"not {in_features} and {out_features}"
            )
        self.register_buffer("elements", torch.empty(out_features, in_features, dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.empty(1, out_features, dtype=torch.int32, device=device))

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Self:
        """``linear`` quantized, kept on its device, its product returned in its weight's dtype."""
        weight = linear.weight
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            out_dtype=weight.dtype,
            device=weight.device,
        )
        with torch.no_grad():
            elements, scales = fp8_rows(weight.detach().to(_work_device(weight)))
            layer.elements.copy_(elements.view(torch.uint8))
            layer.scales.copy_(scales.t().view(torch.int32))
        return layer

    @override
    def forward(self, x: Tensor) -> Tensor:
        rows = self._rows(x)
        elements, scales = quantized_input(fp8_rows, rows)
        out = torch._scaled_mm(
            elements,
            self.elements.view(torch.float8_e4m3fn).t(),
            scale_a=scales,
            scale_b=self.scales.view(torch.float32),
            out_dtype=self.out_dtype,
        )
        return self._shaped(out, x)

    @override
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, out_dtype={self.out_dtype}, fp8_rowwise"
        )


LAYERS: dict[
    DitPrecision, tuple[type[MXFP8Linear] | type[RowwiseFP8Linear], Callable[[int, int], bool], tuple[int, int]]
] = {
    DitPrecision.MXFP8: (MXFP8Linear, fits_mxfp8, BLACKWELL),
    DitPrecision.FP8_ROWWISE: (RowwiseFP8Linear, fits_fp8_rowwise, ADA),
}


def quantize_projections(model: nn.Module, precision: DitPrecision) -> list[str]:
    """Every joint-block projection of ``model`` (:data:`JOINT_BLOCK_LINEARS`) replaced in place by its layer in
    ``precision``; nothing for ``bf16``. Every layer is checked before any is replaced, so a refusal changes nothing.

    Returns:
        The names of the layers replaced, in module order.

    Raises:
        ValueError: If no projection is found, one does not fit the precision, or the GPU cannot multiply in it.
    """
    if precision is DitPrecision.BF16:
        return []
    layer_class, fits, capability = LAYERS[precision]
    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and JOINT_BLOCK_LINEARS.fullmatch(name)
    }
    if not layers:
        raise ValueError(f"no joint-block projection of {type(model).__name__} to compute in {precision}")
    misfits = [name for name, layer in layers.items() if not fits(layer.in_features, layer.out_features)]
    if misfits:
        raise ValueError(f"{len(misfits)} projections do not fit {precision}, the first {misfits[0]}")
    if torch.cuda.is_available() and torch.cuda.get_device_capability() < capability:
        raise ValueError(f"{precision} needs a GPU of compute capability {capability} or later")
    names = list(layers)
    for name in names:  # each bfloat16 layer let go as soon as its replacement is in
        parent, _, child = name.rpartition(".")
        model.get_submodule(parent).register_module(child, layer_class.from_linear(layers.pop(name)))
    return names
