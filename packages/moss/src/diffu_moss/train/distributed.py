"""A full fine-tune over several GPUs: FSDP2 (``torch.distributed.fsdp.fully_shard``), launched by ``torchrun`` on one
node or several.

    torchrun --nproc-per-node 4 -m diffu_moss.train.loop --full --transformer <snapshot>/transformer \\
        --cache data_flux3/cache_moss_pages --out runs/moss_pages_full/full --steps 500 --samples-per-step 8

A process started by ``torchrun`` (its ``WORLD_SIZE`` set) joins the process group (NCCL on CUDA, each rank on its local
GPU; gloo on the CPU) over a one-dimensional mesh of every rank. Each joint block of the DiT is its own FSDP unit and
the rest of the DiT (the embedders, the text refiner, the modulations and the output) the root's: every float32 weight,
its gradient and its optimizer state sharded by rows over the ranks, each unit's weights gathered in bfloat16 for its
forward and backward and its gradients reduced in float32 (:data:`MIXED_PRECISION`); the forward still runs in
bfloat16 autocast, as on one GPU. Without ``torchrun`` nothing here runs: one process trains as before.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from diffu_moss.transformer import MossTransformer

# the weights gathered in bfloat16, the gradients reduced in float32; the inputs left as given (a bfloat16 timestep
# would lose its value: 783 is 784 in bfloat16), autocast casting what the matrix products read
MIXED_PRECISION = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
)


class Ranks(BaseModel):
    """Where this process stands in a run: its rank among ``world`` processes, and their device mesh (None: one
    process, nothing sharded)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    rank: int = 0
    world: int = 1
    mesh: DeviceMesh | None = None

    @property
    def main(self) -> bool:
        """Whether this process writes the run's log and checkpoints (rank 0)."""
        return self.rank == 0

    def owns(self, index: int) -> bool:
        """Whether this rank computes item ``index`` of a list the ranks share out in turn."""
        return index % self.world == self.rank

    def summed(self, values: list[float], device: str) -> list[float]:
        """``values`` summed over the ranks, element by element (each rank giving its own and zeros elsewhere)."""
        if self.mesh is None:
            return values
        total = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(total)
        return total.tolist()


def joined(device: str) -> tuple[Ranks, str]:
    """This process's place in a ``torchrun`` launch and the device it trains on (its local GPU for ``cuda``); outside
    one, one process on ``device``."""
    if "WORLD_SIZE" not in os.environ:
        return Ranks(), device
    kind = torch.device(device).type
    if kind == "cuda":
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
        dist.init_process_group("nccl", device_id=torch.device(device))
    else:
        dist.init_process_group("gloo")
    mesh = init_device_mesh(kind, (dist.get_world_size(),))
    return Ranks(rank=dist.get_rank(), world=dist.get_world_size(), mesh=mesh), device


def left() -> None:
    """The process group left, if this process joined one."""
    if dist.is_initialized():
        dist.destroy_process_group()


def shard(transformer: MossTransformer, mesh: DeviceMesh) -> None:
    """``transformer`` sharded in place over ``mesh`` (the module's docstring): each joint block a unit, then the
    root; its weights moved to the mesh's device."""
    for block in transformer.layers:
        fully_shard(block, mesh=mesh, mp_policy=MIXED_PRECISION)
    fully_shard(transformer, mesh=mesh, mp_policy=MIXED_PRECISION)


def whole(tensor: Tensor) -> Tensor:
    """``tensor`` whole: a sharded one gathered from every rank (a collective), any other as it is."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def sharded_like(full: Tensor, like: Tensor) -> Tensor:
    """``full`` sharded as ``like`` is, on its device and in its dtype, when ``like`` is sharded; else ``full``. Every
    rank holds ``full`` and keeps its own shard, so nothing is sent."""
    if not isinstance(like, DTensor):
        return full
    local = full.to(like.device, like.dtype)
    return distribute_tensor(local, like.device_mesh, list(like.placements), src_data_rank=None)


def gathered(tensors: Mapping[str, Tensor], dtype: torch.dtype, ranks: Ranks) -> dict[str, Tensor]:
    """``tensors`` whole, in ``dtype`` on the CPU, on the main rank (every rank gathering each in turn); empty on the
    others."""
    out = {}
    for name, tensor in tensors.items():
        full = whole(tensor).detach()
        if ranks.main:
            out[name] = full.to("cpu", dtype).contiguous()
    return out


def clip_sharded_gradients(parameters: Iterable[Tensor], max_norm: float) -> float:
    """The sharded gradients of ``parameters`` scaled down to a norm of at most ``max_norm`` (as
    ``torch.nn.utils.clip_grad_norm_`` scales them), and their norm before: the sum of squares of each gradient's rows
    taken on the rank that holds the row, gathered, and summed in one order on every rank, so the norm and the clip are
    the same bits on any number of ranks (torch's sums each rank's shard first, which rounds by the ranks' count).

    Raises:
        ValueError: If a gradient is not sharded (one process: use torch's).
    """
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    rows = []
    for grad in grads:
        if not isinstance(grad, DTensor):
            raise ValueError("clip_sharded_gradients clips sharded gradients: one process clips with torch's")
        local = grad.to_local().float()
        squares = local.reshape(local.shape[0], -1).square().sum(1) if local.ndim > 1 else local.square()
        shape, stride = torch.Size((grad.shape[0],)), (1,)
        rows.append(DTensor.from_local(squares, grad.device_mesh, [Shard(0)], shape=shape, stride=stride).full_tensor())
    norm = torch.cat(rows).double().sum().sqrt().float()
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0).item()
    for grad in grads:
        grad.mul_(scale)
    return norm.item()
