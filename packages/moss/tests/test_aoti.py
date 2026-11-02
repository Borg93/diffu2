"""MOSS-Image's joint block compiled ahead of time (diffu_moss.aoti), on the CPU (AOTInductor compiles for it too):
conftest's tiny DiT run from its package predicts the velocity it predicts eagerly at two token counts, with and without
a reference (the two modulation variants), its projections in their own dtype or in MXFP8; an MXFP8 package takes 129
tokens or more (two tiles of its activations' scales), as its metadata records, refuses fewer and is tuned for no
fewer, and the GPU's MXFP8 block exports for those counts (traced on fake CUDA tensors); and a package built for
another architecture, torch version or precision is refused. The test marked for a Blackwell GPU builds and runs an
MXFP8 package there (cuBLAS's MXFP8 GEMM inside the compiled block) and is skipped without one."""

from __future__ import annotations

import copy
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from diffu_moss import aoti
from diffu_moss.aoti import MAX_TOKENS, Modulation, build_package, export_block, install_package, read_package
from diffu_moss.mxfp8 import DIT_LINEARS, JOINT_BLOCK_LINEARS, Precision, quantize_linears, supports_mxfp8
from diffu_moss.transformer import JointBlock, MossTransformer

# the GPU test's DiT, 128 wide: its projections' MXFP8 scales whole tiles (conftest's tiny DiT on the CPU)
GPU_DIT = {
    "dim": 128,
    "ffn_dim": 256,
    "num_heads": 2,
    "num_layers": 3,
    "text_dim": 128,
    "freq_dim": 16,
    "latent_channels": 16,
}
TUNED_TOKENS = {Precision.BF16: 24, Precision.MXFP8: 140}  # the count each package is tuned for, one it takes
# (rtol, atol) of a packaged velocity to the eager one: float32's own; in MXFP8 the compiled block's arithmetic now and
# then leaves an activation across an E4M3 rounding boundary from its eager value, so that it rounds to the
# neighbouring value and a few elements of the velocity move (11 of 1,984 by up to 7e-5 at 129 tokens, measured)
TOLERANCES = {Precision.BF16: (1.3e-6, 1e-5), Precision.MXFP8: (0.0, 1e-3)}


def _dit(folder: Path, precision: Precision) -> MossTransformer:
    """The DiT saved at ``folder``, in float32, its joint blocks' projections in ``precision``."""
    dit = MossTransformer.from_pretrained(folder).eval()
    if precision is Precision.MXFP8:
        quantize_linears(dit, DIT_LINEARS)
    return dit


def _first_block(dit: MossTransformer) -> JointBlock:
    block = dit.layers[0]
    assert isinstance(block, JointBlock)
    return block


@pytest.fixture(scope="module")
def packages(
    tiny_moss_dit: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[Precision, tuple[MossTransformer, Path]]:
    """Conftest's tiny MOSS DiT in each precision and the package of its joint block, built on the CPU for at most
    4,096 tokens."""
    folder = tmp_path_factory.mktemp("aoti")
    built = {}
    for precision in Precision:
        dit = _dit(tiny_moss_dit, precision)
        package = folder / f"block_{precision}.pt2"
        build_package(_first_block(dit), package, tokens=TUNED_TOKENS[precision], max_tokens=4096)
        built[precision] = (dit, package)
    return built


def _velocity(dit: MossTransformer, grid: tuple[int, int], *, reference: bool) -> torch.Tensor:
    """The DiT's velocity at timestep 700 for latents on ``grid`` and a prompt of 5 tokens, with a 2 x 3 reference
    when asked, all seeded."""
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 16, *grid, generator=generator)
    text = torch.randn(1, 5, 16, generator=generator)
    references = [torch.randn(1, 16, 2, 3, generator=generator)] if reference else None
    with torch.no_grad():
        return dit(latents, torch.tensor([700.0]), text, torch.ones(1, 5, dtype=torch.bool), references=references)


@pytest.mark.parametrize(
    ("precision", "grid"),  # beside the prompt's 5 tokens: counts each package takes, bf16's from 2, MXFP8's from 129
    [
        pytest.param(Precision.BF16, (4, 6), id="bf16-29-tokens"),
        pytest.param(Precision.BF16, (3, 5), id="bf16-20-tokens"),
        pytest.param(Precision.MXFP8, (4, 31), id="mxfp8-129-tokens"),
        pytest.param(Precision.MXFP8, (12, 12), id="mxfp8-149-tokens"),
    ],
)
@pytest.mark.parametrize("reference", [pytest.param(False, id="shared-modulation"), pytest.param(True, id="per-token")])
def test_packaged_dit_predicts_without_its_blocks_own_forward_the_velocity_it_predicts_eagerly(
    packages: dict[Precision, tuple[MossTransformer, Path]],
    precision: Precision,
    grid: tuple[int, int],
    reference: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dit, package = packages[precision]
    eager = _velocity(dit, grid, reference=reference)
    from_package = copy.deepcopy(dit)
    install_package(from_package, package)
    monkeypatch.setattr(JointBlock, "forward", _own_forward_called)

    velocity = _velocity(from_package, grid, reference=reference)

    rtol, atol = TOLERANCES[precision]
    torch.testing.assert_close(velocity, eager, rtol=rtol, atol=atol)


def _own_forward_called(*args: object) -> torch.Tensor:
    raise AssertionError("a joint block ran its own forward, not its package's")


def test_build_package_records_the_token_counts_each_precision_takes(
    packages: dict[Precision, tuple[MossTransformer, Path]],
) -> None:
    recorded = {precision: read_package(package) for precision, (_, package) in packages.items()}

    assert {precision: (info.min_tokens, info.max_tokens) for precision, info in recorded.items()} == {
        Precision.BF16: (2, 4096),
        Precision.MXFP8: (129, 4096),
    }


def test_packaged_dit_refuses_fewer_tokens_than_its_mxfp8_package_takes(
    packages: dict[Precision, tuple[MossTransformer, Path]],
) -> None:
    dit, package = packages[Precision.MXFP8]
    from_package = copy.deepcopy(dit)
    install_package(from_package, package)

    with pytest.raises(ValueError, match=r"takes one sample of 129 to 4096 tokens in float32, not \[1, 128, 32\]"):
        _velocity(from_package, (3, 41), reference=False)  # 123 image tokens and the prompt's 5


def test_build_package_refuses_to_tune_an_mxfp8_package_for_fewer_tokens_than_it_takes(
    tiny_moss_dit: Path, tmp_path: Path
) -> None:
    block = _first_block(_dit(tiny_moss_dit, Precision.MXFP8))

    with pytest.raises(ValueError, match="mxfp8 joint block takes 129 to 4096 tokens, so .* not 128$"):
        build_package(block, tmp_path / "block.pt2", tokens=128, max_tokens=4096)


@contextmanager
def _fake_gpu() -> Generator[None]:
    """Tensors made inside are fake (shapes and dtypes alone: nothing is allocated or computed), in bfloat16 on a CUDA
    device: a GPU's export traced in a process without one."""
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with FakeTensorMode(), torch.device("cuda"):
            yield
    finally:
        torch.set_default_dtype(default)


@pytest.mark.parametrize("modulation", list(Modulation))
def test_export_block_exports_an_mxfp8_block_on_the_gpu_for_129_tokens_and_more(modulation: Modulation) -> None:
    with _fake_gpu():
        block = JointBlock(3072, 8192, 24).eval()  # MOSS-Image's widths
        quantize_linears(block, JOINT_BLOCK_LINEARS)

        program = export_block(block, modulation, tokens=4096, max_tokens=MAX_TOKENS)

    assert torch.ops.aten._scaled_mm_v2.default in {node.target for node in program.graph.nodes}  # not the emulation
    assert [(tokens.lower, tokens.upper) for tokens in program.range_constraints.values()] == [(129, MAX_TOKENS)]


@pytest.mark.parametrize(
    ("elsewhere", "error"),
    [
        pytest.param(
            lambda patch: patch.setattr(aoti, "device_arch", lambda device: "sm_100"),
            "this process runs sm_100 with torch",
            id="architecture",
        ),
        pytest.param(
            lambda patch: patch.setattr(torch, "__version__", "2.99.0"),
            "this process runs .* with torch 2.99.0",
            id="torch-version",
        ),
    ],
)
def test_install_package_refuses_a_package_built_for_another_architecture_or_torch(
    packages: dict[Precision, tuple[MossTransformer, Path]],
    elsewhere: Callable[[pytest.MonkeyPatch], None],
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dit, package = packages[Precision.BF16]
    elsewhere(monkeypatch)

    with pytest.raises(ValueError, match=error):
        install_package(copy.deepcopy(dit), package)


def test_install_package_refuses_a_package_of_another_precision(
    packages: dict[Precision, tuple[MossTransformer, Path]], tiny_moss_dit: Path
) -> None:
    _, package = packages[Precision.BF16]
    quantized = _dit(tiny_moss_dit, Precision.MXFP8)

    with pytest.raises(ValueError, match=r"precision \(bf16 in the package, mxfp8 here\)"):
        install_package(quantized, package)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and supports_mxfp8(torch.device("cuda"))),
    reason="cuBLAS's MXFP8 GEMM needs a GPU of compute capability 10.0 or later (Blackwell)",
)
def test_mxfp8_package_built_on_blackwell_predicts_the_velocity_the_dit_predicts_eagerly(tmp_path: Path) -> None:
    torch.manual_seed(0)
    dit = MossTransformer(**GPU_DIT).eval().cuda().bfloat16()
    quantize_linears(dit, DIT_LINEARS)
    build_package(_first_block(dit), tmp_path / "block.pt2", tokens=512, max_tokens=4096)
    from_package = copy.deepcopy(dit)
    install_package(from_package, tmp_path / "block.pt2")
    similarities = []

    for grid, prompt in (((8, 12), 33), ((16, 24), 40)):  # 129 tokens, the fewest it takes (two tiles of scales); 424
        inputs = (
            torch.randn(1, 16, *grid, device="cuda"),
            torch.tensor([700.0], device="cuda"),
            torch.randn(1, prompt, 128, device="cuda", dtype=torch.bfloat16),
            torch.ones(1, prompt, dtype=torch.bool, device="cuda"),
        )
        with torch.no_grad():
            eager, velocity = dit(*inputs).float(), from_package(*inputs).float()
        similarities.append(torch.nn.functional.cosine_similarity(velocity.flatten(), eager.flatten(), dim=0).item())

    assert min(similarities) > 0.999
