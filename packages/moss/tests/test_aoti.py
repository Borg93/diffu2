"""MOSS-Image's joint block compiled ahead of time (diffu_moss.aoti), on the CPU (AOTInductor compiles for it too):
conftest's tiny DiT run from its package predicts the velocity it predicts eagerly at two token counts, with and without
a reference (the two modulation variants), its projections in their own dtype or in MXFP8; and a package built for
another architecture, torch version or precision is refused. The test marked for a Blackwell GPU builds and runs an
MXFP8 package there (cuBLAS's MXFP8 GEMM inside the compiled block) and is skipped without one."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from diffu_moss import aoti
from diffu_moss.aoti import build_package, install_package
from diffu_moss.mxfp8 import DIT_LINEARS, Precision, quantize_linears, supports_mxfp8
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
        build_package(_first_block(dit), folder / f"block_{precision}.pt2", tokens=24, max_tokens=4096)
        built[precision] = (dit, folder / f"block_{precision}.pt2")
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


@pytest.mark.parametrize("precision", list(Precision))
@pytest.mark.parametrize("grid", [pytest.param((4, 6), id="29-tokens"), pytest.param((3, 5), id="20-tokens")])
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

    torch.testing.assert_close(velocity, eager)


def _own_forward_called(*args: object) -> torch.Tensor:
    raise AssertionError("a joint block ran its own forward, not its package's")


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
    inputs = (
        torch.randn(1, 16, 16, 24, device="cuda"),
        torch.tensor([700.0], device="cuda"),
        torch.randn(1, 40, 128, device="cuda", dtype=torch.bfloat16),
        torch.ones(1, 40, dtype=torch.bool, device="cuda"),
    )
    with torch.no_grad():
        eager = dit(*inputs).float()

        velocity = from_package(*inputs).float()

    assert torch.nn.functional.cosine_similarity(velocity.flatten(), eager.flatten(), dim=0) > 0.999
