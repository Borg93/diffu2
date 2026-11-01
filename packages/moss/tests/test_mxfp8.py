"""MXFP8 (diffu_moss.mxfp8) against references written from its specifications: each block's E8M0 scale and its E4M3
elements as the OCP MX v1.0 spec's conversion gives them (section 6.3; the E4M3 values from its table 2, rounded to
nearest, ties to even, saturating), the scale layout cuBLAS documents for block-scaled GEMMs, a quantized layer's
product, the layers each model's swap picks, the GPU path traced as one graph whatever the token count and its operands
as scaled_mm checks them (on the CPU, up to the kernel it lacks), and the GPUs that multiply MXFP8. The tests marked for
a Blackwell GPU run cuBLAS's MXFP8 GEMM itself and are skipped without one."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch._dynamo.testing import CompileCounter
from transformers import Qwen3_5Config, Qwen3_5Model

from diffu_moss import mxfp8
from diffu_moss.mxfp8 import (
    DIT_LINEARS,
    JOINT_BLOCK_LINEARS,
    TEXT_ENCODER_LINEARS,
    block_scaled_mm,
    dequantize,
    quantize,
    quantize_linears,
    supports_mxfp8,
    swizzled,
    swizzled_size,
)
from diffu_moss.transformer import JointBlock, MossTransformer

BLACKWELL = pytest.mark.skipif(
    not (torch.cuda.is_available() and supports_mxfp8(torch.device("cuda"))),
    reason="cuBLAS's MXFP8 GEMM needs a GPU of compute capability 10.0 or later (Blackwell)",
)
TINY_DIT = {
    "dim": 32,
    "ffn_dim": 64,
    "num_heads": 2,
    "num_layers": 2,
    "text_dim": 16,
    "freq_dim": 16,
    "latent_channels": 16,
}


def e4m3_magnitudes() -> list[float]:
    """Every finite non-negative E4M3 value in code order, by the OCP FP8 encoding (bias 7, 3 mantissa bits, no
    infinities, S.1111.111 the NaN): code ``8 * exponent + mantissa``."""
    return [
        2.0**-6 * (code & 7) / 8 if code >> 3 == 0 else 2.0 ** ((code >> 3) - 7) * (1 + (code & 7) / 8)
        for code in range(0x7F)
    ]


E4M3 = e4m3_magnitudes()


def nearest_e4m3(magnitude: float) -> float:
    """``magnitude`` rounded to the nearest E4M3 value, a tie to the even code, past 448 saturated to it."""
    if magnitude >= E4M3[-1]:
        return E4M3[-1]
    upper = next(code for code, value in enumerate(E4M3) if value >= magnitude)
    if E4M3[upper] == magnitude:
        return magnitude
    below, above = magnitude - E4M3[upper - 1], E4M3[upper] - magnitude
    if below == above:
        return E4M3[upper] if upper % 2 == 0 else E4M3[upper - 1]
    return E4M3[upper - 1] if below < above else E4M3[upper]


def spec_block(block: list[float]) -> tuple[int, list[float]]:
    """A block of 32 as the MX spec converts it (section 6.3): its scale's E8M0 byte (the largest power of two at most
    its largest magnitude over E4M3's 2^8, held within E8M0's 2^-127 to 2^127; 255, the NaN, for a block with an
    infinity or a NaN) and the values its elements stand for."""
    if not all(math.isfinite(value) for value in block):
        return 255, [math.nan] * len(block)
    amax = max(abs(value) for value in block)
    exponent = math.frexp(amax)[1] - 1 if amax else -127  # floor(log2(amax))
    byte = min(max(exponent - 8 + 127, 0), 254)
    scale = 2.0 ** (byte - 127)
    return byte, [math.copysign(nearest_e4m3(abs(value) / scale), value) * scale for value in block]


def _block(values: list[float]) -> list[float]:
    """``values`` repeated to a block of 32."""
    return (values * 32)[:32]


ADVERSARIAL_BLOCKS = [
    pytest.param(torch.randn(4, 32, generator=torch.Generator().manual_seed(0)).tolist(), id="normal"),
    pytest.param(
        (
            torch.randn(2, 32, generator=torch.Generator().manual_seed(1)) * 2.0 ** torch.arange(-20, 44).view(2, 32)
        ).tolist(),
        id="spread-over-2^64",
    ),
    pytest.param([[0.0] * 32], id="zeros"),
    pytest.param([[1e-3, -2e-3, 3e-4] * 10 + [5e-4, 1e4]], id="one-outlier"),
    pytest.param([_block([1e-40, -3e-41, 7e-39, 0.0])], id="float32-subnormals"),
    pytest.param([_block([256.0, 1.0625, 1.1875, -0.0029296875])], id="power-of-two-largest-and-ties"),
    pytest.param([_block([255.99998, 1.0, -0.5])], id="largest-just-below-a-power-of-two"),
    pytest.param([_block([509.0, 476.0, -464.0, 300.0])], id="saturating"),
    pytest.param([_block([3.4028234663852886e38, -1e38, 1.0])], id="largest-float32"),
    pytest.param([_block([1.0, math.inf, 2.0])], id="infinity"),
    pytest.param([_block([1.0, math.nan, 2.0])], id="nan"),
]


@pytest.mark.parametrize("blocks", ADVERSARIAL_BLOCKS)
def test_quantize_scales_and_rounds_each_block_as_the_mx_spec_converts_it(blocks: list[list[float]]) -> None:
    values = torch.tensor(blocks, dtype=torch.float32)

    quantized = quantize(values)

    expected = [spec_block(block) for block in values.tolist()]
    assert quantized.scales.tolist() == [[byte] for byte, _ in expected]
    read_back = dequantize(quantized)
    expected_values = torch.tensor([block for _, block in expected], dtype=torch.float32)
    torch.testing.assert_close(read_back, expected_values, rtol=0, atol=0, equal_nan=True)
    for block, (byte, _), back in zip(values.tolist(), expected, read_back.tolist(), strict=True):
        scale = 2.0 ** (byte - 127)  # within half an E4M3 step: 1/16 of a normal value, 2^-10 of the scale below them
        unsaturated = [(value, near) for value, near in zip(block, back, strict=True) if abs(value) <= 448 * scale]
        if byte != 255:
            assert all(abs(near - value) <= max(abs(value) / 16, scale / 1024) for value, near in unsaturated)


def test_swizzled_lays_out_block_scales_where_cublas_reads_them() -> None:
    rows, blocks = 200, 7  # two tiles of rows (the second padded), two of blocks (the second padded)
    scales = torch.arange(1, rows * blocks + 1, dtype=torch.int32).view(rows, blocks)

    laid_out = swizzled(scales)

    # cuBLAS's 1D block scaling factors layout: 128x4 tiles, row-major, (r % 32) * 16 + (r % 128 // 32) * 4 + c % 4
    expected = torch.zeros(swizzled_size(rows, blocks), dtype=torch.int32)
    for row in range(rows):
        for block in range(blocks):
            tile = row // 128 * 2 + block // 4
            expected[tile * 512 + row % 32 * 16 + row % 128 // 32 * 4 + block % 4] = scales[row, block]
    assert torch.equal(laid_out, expected)


def _varied(rows: int, columns: int, seed: int) -> torch.Tensor:
    """Values whose blocks of 32 differ in magnitude by up to 2^±8, so a scale read from the wrong place shows."""
    generator = torch.Generator().manual_seed(seed)
    blocks = 2.0 ** torch.randint(-8, 8, (rows, columns // 32, 1), generator=generator)
    return (torch.randn(rows, columns // 32, 32, generator=generator) * blocks).flatten(1)


def test_mxfp8_linear_computes_the_product_of_its_quantized_input_and_weight_plus_its_bias() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(96, 144)  # 3 blocks of inputs (one tile, padded); 144 outputs (two tiles, the second padded)
    with torch.no_grad():
        linear.weight.copy_(_varied(144, 96, 0))
    x = _varied(15, 96, 1).view(3, 5, 96)
    layer = mxfp8.Linear.from_linear(linear)

    out = layer(x)

    weight = dequantize(quantize(linear.weight.detach()))
    expected = (dequantize(quantize(x.reshape(15, 96))) @ weight.t()).view(3, 5, 144) + linear.bias.detach()
    assert torch.equal(out, expected)


def _precisions(model: nn.Module) -> dict[str, str]:
    """Every linear layer of ``model`` by name: ``mxfp8`` or ``bf16`` (its own weight)."""
    return {
        name: "mxfp8" if isinstance(module, mxfp8.Linear) else "bf16"
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear | mxfp8.Linear)
    }


def test_quantize_linears_swaps_exactly_the_dits_joint_block_projections() -> None:
    dit = MossTransformer(**TINY_DIT)
    linears = _precisions(dit)

    swapped = quantize_linears(dit, DIT_LINEARS)

    projections = [
        f"layers.{k}.proj.{name}" for k in (0, 1) for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
    ]
    assert swapped == projections
    assert _precisions(dit) == {name: "mxfp8" if name in projections else "bf16" for name in linears}


def test_quantize_linears_swaps_every_text_encoder_decoder_matrix_but_the_delta_rules_gates() -> None:
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 64,
        },
    )
    encoder = Qwen3_5Model(config)
    linears = _precisions(encoder)

    swapped = quantize_linears(encoder, TEXT_ENCODER_LINEARS)

    mlp = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    delta_rule = ["linear_attn.out_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]  # not in_proj_a, _b
    attention = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    expected = [f"language_model.layers.0.{name}" for name in delta_rule + mlp]
    expected += [f"language_model.layers.1.{name}" for name in attention + mlp]
    assert swapped == expected
    assert _precisions(encoder) == {name: "mxfp8" if name in expected else "bf16" for name in linears}


@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param(
            {**TINY_DIT, "ffn_dim": 48}, "2 chosen layers are not, the first layers.0.proj.ffn_down", id="misfit"
        ),
        pytest.param({**TINY_DIT, "num_layers": 0}, "no linear layer of MossTransformer", id="none-named"),
    ],
)
def test_quantize_linears_refuses_before_swapping_any_layer(config: dict[str, int], error: str) -> None:
    dit = MossTransformer(**config)
    linears = _precisions(dit)

    with pytest.raises(ValueError, match=error):
        quantize_linears(dit, DIT_LINEARS)

    assert _precisions(dit) == linears


def test_block_scaled_mm_traces_as_one_graph_whatever_the_token_count() -> None:
    counter = CompileCounter()  # counts the graphs compiled: a graph break or a recompile for a new count adds one
    compiled = torch.compile(block_scaled_mm, backend=counter, dynamic=True)
    weight = torch.empty(384, 256, dtype=torch.float8_e4m3fn, device="meta")
    scales = torch.empty(swizzled_size(384, 8), dtype=torch.uint8, device="meta")

    shapes = [
        compiled(torch.empty(tokens, 256, dtype=torch.bfloat16, device="meta"), weight, scales, torch.bfloat16).shape
        for tokens in (200, 4133)
    ]

    assert shapes == [(200, 384), (4133, 384)]
    assert counter.frame_count == 1


def test_block_scaled_mm_gives_scaled_mm_mxfp8_operands_it_takes_wanting_only_a_gpu() -> None:
    rows = _varied(200, 96, 0).bfloat16()  # each operand's scales padded, in rows (200, 144) and blocks (3)
    weight = quantize(_varied(144, 96, 1))

    # scaled_mm checks its operands (their dtypes, the scales' recipe, layout and count) before it picks a kernel, and
    # the CPU has none for swizzled MX scales; once it has, this compares the product with the emulated one instead
    with pytest.raises(ValueError, match="CPU does not support swizzle"):
        block_scaled_mm(rows, weight.elements, swizzled(weight.scales), torch.bfloat16)


@pytest.mark.parametrize(
    ("capability", "native"),
    [pytest.param((9, 0), False, id="hopper"), pytest.param((10, 0), True, id="blackwell")],
)
def test_supports_mxfp8_from_compute_capability_10(
    capability: tuple[int, int], native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)

    assert supports_mxfp8(torch.device("cuda")) is native


@BLACKWELL
def test_block_scaled_mm_on_blackwell_equals_the_product_emulated_from_the_same_blocks() -> None:
    rows = _varied(200, 256, 0).cuda().bfloat16()  # 200 rows: the second tile of scales padded
    weight = quantize(_varied(384, 256, 1).cuda())

    product = block_scaled_mm(rows, weight.elements, swizzled(weight.scales), torch.bfloat16)

    emulated = dequantize(quantize(rows)) @ dequantize(weight).t()
    torch.testing.assert_close(product.float(), emulated, rtol=2**-7, atol=1e-4 * emulated.abs().max().item())


@BLACKWELL
def test_mxfp8_joint_block_compiled_on_blackwell_computes_as_it_does_eagerly_at_two_token_counts() -> None:
    torch.manual_seed(0)
    block = JointBlock(128, 256, 2).cuda().bfloat16().eval()
    quantize_linears(block, JOINT_BLOCK_LINEARS)
    compiled = torch.compile(block, fullgraph=True, dynamic=True)
    similarities = []

    for tokens in (300, 1000):
        x = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
        rope = torch.randn(1, tokens, 64, device="cuda", dtype=torch.bfloat16)
        modulation = torch.randn(1, 6, 128, device="cuda", dtype=torch.bfloat16) * 0.1
        with torch.no_grad():
            eager, fused = block(x, rope, modulation).float(), compiled(x, rope, modulation).float()
        similarities.append(torch.nn.functional.cosine_similarity(fused.flatten(), eager.flatten(), dim=0).item())

    assert min(similarities) > 0.999
