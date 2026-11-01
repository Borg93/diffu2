"""diffu-moss-aoti: MOSS-Image's joint block compiled ahead of time (AOTInductor) into a package, so a worker draws
with compiled blocks without compiling: a package is built once per GPU architecture and torch version, and loading
one takes seconds.

    diffu-moss-aoti --model runs/moss/base --precision mxfp8 --out runs/aoti/moss_mxfp8_sm100.pt2

One block is exported (``torch.export``) as a function of its own tensors, its token count dynamic (pages differ in
size), so the one compiled block serves all 42: each block hands it its own weights at every call, and the package
holds no weights. It holds two compiled variants of the block, one per form of its time modulation: shared by every
token (``[1, 6, dim]``: a page drawn without a reference) or one per token (``[1, tokens, 6, dim]``: a reference's
tokens at timestep 0). A package is built for one precision of the blocks' projections (bf16, or MXFP8:
:mod:`diffu_moss.mxfp8`), the block's widths and dtype, and is valid only on the GPU architecture and with the torch
version that built it: all are recorded in its metadata, and a package that does not match is refused before its code
is loaded.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Protocol

import torch
import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from torch import Tensor, nn
from torch._inductor import aot_compile
from torch._inductor.package import load_package, package_aoti
from torch.export import Dim, ExportedProgram
from torch.export.pt2_archive import PT2ArchiveReader
from torch.export.pt2_archive.constants import AOTINDUCTOR_DIR
from torch.func import functional_call

from diffu_core.repo import ModelIndex
from diffu_moss import mxfp8
from diffu_moss.mxfp8 import JOINT_BLOCK_LINEARS, Precision, quantize_linears, supports_mxfp8
from diffu_moss.transformer import MODULATIONS, JointBlock, MossTransformer

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

PACKAGE_FORMAT = 1
METADATA_KEY = "diffu_moss.joint_block"  # the package metadata's entry holding a BlockPackage as JSON
MIN_TOKENS = 2  # an image token and a prompt token at least
MAX_TOKENS = 1 << 16  # a page of ~6 MP with its prompt and sample; int32 indexing holds the widest product of it
EXAMPLE_TOKENS = 4096  # a 1 MP page's image tokens: the size the compiler tunes for


class Modulation(StrEnum):
    """The two forms of a joint block's time modulation, a compiled variant each."""

    SHARED = "shared"  # [1, 6, dim]: every token at the flow timestep
    PER_TOKEN = "per_token"  # [1, tokens, 6, dim]: a reference's tokens at timestep 0, the rest at the flow timestep


class BlockPackage(BaseModel):
    """What a package of the joint block was built for, as its metadata records it.

    Attributes:
        format: The package's layout, :data:`PACKAGE_FORMAT`.
        torch: The torch version that built it.
        arch: The device architecture it was built for (:func:`device_arch`).
        precision: The precision of the block's projections.
        dtype: The dtype of the block's activations (``bfloat16``).
        dim: The block's width.
        ffn_dim: Its feed-forward width.
        heads: Its attention heads.
        tensors: The block's tensors by name, in the order the compiled block takes them.
        max_tokens: The most tokens it takes.
    """

    model_config = ConfigDict(frozen=True)

    format: int = PACKAGE_FORMAT
    torch: str
    arch: str
    precision: Precision
    dtype: str
    dim: int
    ffn_dim: int
    heads: int
    tensors: tuple[str, ...]
    max_tokens: int = Field(ge=MIN_TOKENS)


class CompiledBlock(Protocol):
    """A loaded package's compiled block: the joint block's output for its input and its tensors."""

    def __call__(self, x: Tensor, rope: Tensor, modulation: Tensor, tensors: list[Tensor]) -> Tensor: ...


def device_arch(device: torch.device) -> str:
    """The architecture a package for ``device`` is compiled for: a CUDA GPU's compute capability (``sm_100``), or the
    CPU's machine and vector extension (``cpu-aarch64-DEFAULT``)."""
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return f"sm_{major}{minor}"
    return f"{device.type}-{platform.machine()}-{torch.backends.cpu.get_cpu_capability()}"


def dtype_name(dtype: torch.dtype) -> str:
    """A dtype's name as a package records it (``bfloat16``)."""
    return str(dtype).removeprefix("torch.")


def block_tensors(block: nn.Module) -> dict[str, Tensor]:
    """A block's parameters and buffers by name: what its compiled form takes at every call."""
    return dict(block.named_parameters()) | dict(block.named_buffers())


def block_precision(block: JointBlock) -> Precision:
    """The precision of a joint block's projections."""
    return Precision.MXFP8 if isinstance(block.proj["q"], mxfp8.Linear) else Precision.BF16


def block_description(block: JointBlock, *, max_tokens: int) -> BlockPackage:
    """What a package of ``block`` built on its device by this process is."""
    norm = block.norm1.weight
    ffn_down = block.proj["ffn_down"]
    if not isinstance(ffn_down, nn.Linear | mxfp8.Linear):
        raise TypeError(f"the block's ffn_down is a {type(ffn_down).__name__}, not a linear layer")
    return BlockPackage(
        torch=torch.__version__,
        arch=device_arch(norm.device),
        precision=block_precision(block),
        dtype=dtype_name(norm.dtype),
        dim=norm.shape[0],
        ffn_dim=ffn_down.in_features,
        heads=block.heads,
        tensors=tuple(block_tensors(block)),
        max_tokens=max_tokens,
    )


class TensorsCall(nn.Module):
    """A joint block as a function of its own tensors: what is exported, so that one compiled block serves every
    block. The block it calls is kept out of this module's state (in a list): the call replaces every tensor of it
    (``functional_call``), so none is captured into the package."""

    def __init__(self, block: JointBlock) -> None:
        super().__init__()
        self.names = tuple(block_tensors(block))
        self.blocks = [block]

    def forward(  # ty: ignore[invalid-mutable-override]  torch declares forward an attribute; Modules override it
        self,
        x: Tensor,
        rope: Tensor,
        modulation: Tensor,
        tensors: list[Tensor],
    ) -> Tensor:
        own = dict(zip(self.names, tensors, strict=True))
        return functional_call(self.blocks[0], own, (x, rope, modulation), strict=True)


def export_block(block: JointBlock, modulation: Modulation, *, tokens: int, max_tokens: int) -> ExportedProgram:
    """``block`` exported as a function of its tensors (:class:`TensorsCall`) for one sample of 2 to ``max_tokens``
    tokens, its modulation of the form ``modulation``; ``tokens`` is the example's count."""
    norm = block.norm1.weight
    dim = norm.shape[0]
    shared = modulation is Modulation.SHARED
    x = torch.randn(1, tokens, dim, device=norm.device, dtype=norm.dtype)
    rope = torch.randn(1, tokens, dim // block.heads, device=norm.device, dtype=norm.dtype)
    modulation_shape = (1, MODULATIONS, dim) if shared else (1, tokens, MODULATIONS, dim)
    modulation_input = torch.randn(modulation_shape, device=norm.device, dtype=norm.dtype)
    tensors = [tensor.detach() for tensor in block_tensors(block).values()]
    count = Dim("tokens", min=MIN_TOKENS, max=max_tokens)
    dynamic = {
        "x": {1: count},
        "rope": {1: count},
        "modulation": None if shared else {1: count},
        "tensors": [None] * len(tensors),
    }
    with torch.no_grad():
        return torch.export.export(
            TensorsCall(block), (x, rope, modulation_input, tensors), dynamic_shapes=dynamic, strict=False
        )


def build_package(
    block: JointBlock, out: Path, *, tokens: int = EXAMPLE_TOKENS, max_tokens: int = MAX_TOKENS
) -> BlockPackage:
    """``block`` compiled by AOTInductor for its device, in both modulation variants, and packaged into ``out`` (a
    new ``.pt2`` file) with what it was built for in its metadata.

    Returns:
        What the package was built for.

    Raises:
        ValueError: If ``out`` is not a new ``.pt2`` file.
    """
    if out.suffix != ".pt2" or out.exists():
        raise ValueError(f"{out} is not a new .pt2 file")
    info = block_description(block, max_tokens=max_tokens)
    options = {"aot_inductor.package": True, "aot_inductor.metadata": {METADATA_KEY: info.model_dump_json()}}
    files = {}
    for modulation in Modulation:
        program = export_block(block, modulation, tokens=tokens, max_tokens=max_tokens)
        if program.example_inputs is None:
            raise TypeError("AOTInductor compiles an exported program from its example inputs, and this has none")
        args, kwargs = program.example_inputs
        compiled = aot_compile(program.module(check_guards=False), args, kwargs, options=options)
        if not isinstance(compiled, list):
            raise TypeError(f"AOTInductor gave {type(compiled).__name__}, not the package's files")
        files[modulation.value] = compiled
    out.parent.mkdir(parents=True, exist_ok=True)
    package_aoti(str(out), files)
    return info


def read_package(path: Path) -> BlockPackage:
    """What the package at ``path`` was built for, read from its metadata without loading its code.

    Raises:
        ValueError: If ``path`` is not a package of the joint block.
    """
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    with PT2ArchiveReader(str(path)) as archive:
        prefix = f"{AOTINDUCTOR_DIR}{Modulation.SHARED.value}/"
        for name in archive.get_file_names():
            if name.startswith(prefix) and name.endswith("_metadata.json"):
                metadata = json.loads(archive.read_string(name))
                if METADATA_KEY in metadata:
                    return BlockPackage.model_validate_json(metadata[METADATA_KEY])
    raise ValueError(f"{path} is not a package of MOSS-Image's joint block (diffu-moss-aoti)")


def check_package(info: BlockPackage, device: torch.device) -> None:
    """Refuse a package of another format, or built for another device architecture or by another torch version than
    ``device``'s and this process's.

    Raises:
        ValueError: If any of them differs.
    """
    here = (PACKAGE_FORMAT, device_arch(device), torch.__version__)
    if (info.format, info.arch, info.torch) != here:
        raise ValueError(
            f"the package was built for {info.arch} by torch {info.torch} (format {info.format}); this process runs "
            f"{here[1]} with torch {here[2]} (format {here[0]}): build one here with diffu-moss-aoti"
        )


class PackagedForward:
    """A joint block's forward run by a package's compiled block, handed the block's own tensors at every call (so the
    block is read as it is, wherever it was moved since)."""

    def __init__(self, block: JointBlock, compiled: Mapping[Modulation, CompiledBlock], info: BlockPackage) -> None:
        self.block = block
        self.compiled = compiled
        self.info = info

    def __call__(self, x: Tensor, rope: Tensor, modulation: Tensor) -> Tensor:
        """The block's output for ``x`` ``[1, tokens, dim]``, as its own forward takes it.

        Raises:
            ValueError: If ``x`` is not one sample of 2 to the package's most tokens, in its dtype.
        """
        batch, tokens = x.shape[:2]
        if batch != 1 or not MIN_TOKENS <= tokens <= self.info.max_tokens or dtype_name(x.dtype) != self.info.dtype:
            raise ValueError(
                f"a packaged joint block takes one sample of {MIN_TOKENS} to {self.info.max_tokens} tokens in "
                f"{self.info.dtype}, not {list(x.shape)} in {x.dtype}"
            )
        form = Modulation.SHARED if modulation.dim() == 3 else Modulation.PER_TOKEN
        tensors = list(block_tensors(self.block).values())
        return self.compiled[form](x.contiguous(), rope.contiguous(), modulation.contiguous(), tensors)


def install_package(transformer: MossTransformer, path: Path) -> BlockPackage:
    """Every joint block of ``transformer`` run by the package at ``path`` from now on, the package checked against
    this process and the transformer (its device, precision, widths, dtype and every block's tensors) before its code
    is loaded.

    Returns:
        What the package was built for.

    Raises:
        ValueError: If the package does not match.
    """
    info = read_package(path)
    blocks = [block for block in transformer.layers if isinstance(block, JointBlock)]
    device = blocks[0].norm1.weight.device
    check_package(info, device)
    here = block_description(blocks[0], max_tokens=info.max_tokens)
    differences = [
        f"{field} ({getattr(info, field)} in the package, {getattr(here, field)} here)"
        for field in BlockPackage.model_fields
        if getattr(info, field) != getattr(here, field)
    ]
    if differences:
        raise ValueError(f"the package does not fit this transformer: {'; '.join(differences)}")
    if any(tuple(block_tensors(block)) != info.tensors for block in blocks):
        raise ValueError("the transformer's joint blocks do not all hold the tensors the package takes")
    index = -1 if device.index is None else device.index
    compiled: dict[Modulation, CompiledBlock] = {
        modulation: load_package(str(path), modulation.value, device_index=index) for modulation in Modulation
    }
    for block in blocks:
        block.forward = PackagedForward(block, compiled, info)
    return info


def joint_block(model: Path, precision: Precision, device: torch.device) -> JointBlock:
    """A joint block of the MOSS-Image repo ``model``'s DiT, by its config (its weights random: a package holds none),
    in bfloat16 on ``device``, its projections in ``precision``.

    Raises:
        ValueError: If MXFP8 is asked of a GPU without it.
    """
    if precision is Precision.MXFP8 and device.type == "cuda" and not supports_mxfp8(device):
        raise ValueError(f"MXFP8 needs a GPU of compute capability 10.0 or later (Blackwell), not {device}")
    config = MossTransformer.load_config(ModelIndex.read(model).source("transformer").repo)
    block = JointBlock(config["dim"], config["ffn_dim"], config["num_heads"]).to(device, torch.bfloat16)
    if precision is Precision.MXFP8:
        quantize_linears(block, JOINT_BLOCK_LINEARS)
    return block.eval()


@app.command()
def main(
    model: Annotated[Path, typer.Option(help="a MOSS-Image model directory (diffu-moss-init's): its DiT's widths")],
    out: Annotated[Path, typer.Option(help="the package to write (a new .pt2 file)")],
    precision: Annotated[Precision, typer.Option(help="the precision of the blocks' projections")] = Precision.BF16,
    device: Annotated[
        str, typer.Option(help="the device to build for: the package runs on its architecture only")
    ] = "cuda",
    tokens: Annotated[
        int, typer.Option(min=MIN_TOKENS, help="the token count the compiler tunes for")
    ] = EXAMPLE_TOKENS,
) -> None:
    """Compile MOSS-Image's joint block ahead of time into a package the page pipeline loads (aoti_package)."""
    try:
        info = build_package(joint_block(model, precision, torch.device(device)), out, tokens=tokens)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"the joint block in {info.precision}, for {info.arch} and torch {info.torch} -> {out}")


if __name__ == "__main__":
    app()
