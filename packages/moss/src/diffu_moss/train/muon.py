"""Muon as MOSS-Image's technical report steps its joint blocks' matrices (Appendix A.2, Table 7): a fused matrix
orthogonalised block by block, each of its row blocks on its own.

Each matrix takes Nesterov momentum (``buffer = lerp(buffer, gradient, 1 - momentum)``, the update ``lerp(gradient,
buffer, momentum)``), the update orthogonalised by 5 quintic Newton-Schulz steps in bfloat16 (Keller Jordan's
coefficients 3.4445, -4.7750, 2.0315), decoupled weight decay (``w *= 1 - lr * decay``) and the step scaled to
AdamW's RMS (``lr * 0.2 * sqrt(max(rows, cols))``, Moonshot's "Muon is Scalable for LLM Training"). This is exactly
``torch.optim.Muon`` with ``adjust_lr_fn="match_rms_adamw"``, but a matrix of ``splits`` row blocks (the gated
feed-forward's ``ffn_up_gate``: its value half, then its gate half) is orthogonalised and RMS-scaled one block at a
time, as the separate matrices the fused one stands for; the parameter stays one tensor, so the DiT's weights keep
their names.

The momentum is kept in float32, as torch's Muon keeps it, or with ``momentum_dtype=torch.bfloat16`` in bfloat16, for
half its memory (9.5 GB less for the real DiT's 4.76 B joint-block weights). A bfloat16 momentum is read in float32 and
moved towards the gradient there (the same ``lerp``); the update is taken from that float32 momentum, and the momentum
is stored back rounded to bfloat16 (to nearest even). The update is cast to bfloat16 for the Newton-Schulz steps either
way, so the only difference is the rounding of the momentum carried to the next step. A checkpoint's state keeps it in
bfloat16, and a resumed run reads it back so (:meth:`Muon.load_state_dict`).

A matrix sharded by FSDP2 (:mod:`diffu_moss.train.distributed`) takes its momentum and decay on each rank's rows. Its
update is orthogonalised by one rank, its owner (the ``k``-th sharded matrix of a step is rank ``k % world``'s): every
rank sends its rows of the update to the owner (in bfloat16, which the Newton-Schulz steps cast it to first), the owner
orthogonalises it whole, and sends each rank its rows back; every gather of a step is under way before any owner
computes, and every owner computes its matrices before the rows go back. That is exactly the step of one process, the
same bits, with each rank orthogonalising a world's share of the matrices (the real DiT's ~3.7e14 FLOPs a step, ~1.5 s
on an A100, spread over the ranks) and sending its share of the updates once, not gathering all 9.5 GB of them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)
NEWTON_SCHULZ_STEPS = 5
NORM_EPS = 1e-7
RMS_MATCH = 0.2
MOMENTUM_BUFFER = "momentum_buffer"


def orthogonalised(update: Tensor, steps: int = NEWTON_SCHULZ_STEPS) -> Tensor:
    """The nearest semi-orthogonal matrix to ``update`` (2-D), approximately: ``steps`` quintic Newton-Schulz steps in
    bfloat16 on it scaled to a spectral norm of at most one (in bfloat16, as torch's Muon returns it)."""
    a, b, c = NEWTON_SCHULZ
    tall = update.size(0) > update.size(1)
    x = update.bfloat16()
    if tall:
        x = x.T
    x = x / x.norm().clamp(min=NORM_EPS)
    for _ in range(steps):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return x.T if tall else x


def rms_matched_rate(learning_rate: float, rows: int, cols: int) -> float:
    """The learning rate of an orthogonalised ``rows`` x ``cols`` update whose RMS matches AdamW's."""
    return learning_rate * RMS_MATCH * math.sqrt(max(rows, cols))


class Muon(Optimizer):
    """Muon over 2-D matrices (the module's docstring); a group's ``splits`` is how many row blocks each of its
    matrices is orthogonalised in (1: whole).

    Attributes:
        momentum_dtype: The dtype every momentum is kept in: float32, or bfloat16 for half the memory.

    Raises:
        ValueError: From the constructor, if a parameter is not a matrix or its rows do not split into ``splits``.
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool = True,
        ns_steps: int = NEWTON_SCHULZ_STEPS,
        splits: int = 1,
        momentum_dtype: torch.dtype = torch.float32,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "splits": splits,
        }
        super().__init__(params, defaults)
        self.momentum_dtype = momentum_dtype
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2 or parameter.shape[0] % group["splits"]:
                    raise ValueError(
                        f"Muon steps matrices whose rows split into {group['splits']}, not {list(parameter.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure: None = None) -> None:  # ty: ignore[invalid-method-override]  no closure: one loss a step
        """Every matrix with a gradient stepped (the module's docstring): a sharded one's update orthogonalised by its
        owner rank, any other's here."""
        sharded: list[tuple[DTensor, DTensor, dict[str, Any]]] = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = self._momentum_update(parameter, parameter.grad, group)
                if isinstance(parameter, DTensor) and isinstance(update, DTensor):
                    sharded.append((parameter, update, group))
                else:
                    self._apply(parameter, self._orthogonal(update.bfloat16(), group), group)
        if sharded:
            self._step_sharded(sharded)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """``state_dict`` loaded, every momentum in :attr:`momentum_dtype`: torch casts each floating-point state to its
        parameter's dtype (float32), so a bfloat16 momentum is cast back, exactly."""
        super().load_state_dict(state_dict)
        for state in self.state.values():
            if MOMENTUM_BUFFER in state:
                state[MOMENTUM_BUFFER] = state[MOMENTUM_BUFFER].to(self.momentum_dtype)

    def _momentum_update(self, parameter: Tensor, grad: Tensor, group: dict[str, Any]) -> Tensor:
        """The parameter's momentum moved towards ``grad`` and its (Nesterov) update, on this rank's rows when sharded."""
        state = self.state[parameter]
        if MOMENTUM_BUFFER not in state:
            state[MOMENTUM_BUFFER] = torch.zeros_like(
                grad, dtype=self.momentum_dtype, memory_format=torch.preserve_format
            )
        buffer = state[MOMENTUM_BUFFER]
        if buffer.dtype == torch.float32:
            momentum = buffer.lerp_(grad, 1 - group["momentum"])
        else:  # read in float32, stored back rounded; the update takes it unrounded
            momentum = buffer.float().lerp_(grad, 1 - group["momentum"])
            buffer.copy_(momentum)
        return grad.lerp(momentum, group["momentum"]) if group["nesterov"] else momentum

    @staticmethod
    def _orthogonal(update: Tensor, group: dict[str, Any]) -> Tensor:
        """A whole matrix's update orthogonalised, each of its ``splits`` row blocks on its own."""
        blocks = [orthogonalised(block, group["ns_steps"]) for block in update.chunk(group["splits"], 0)]
        return torch.cat(blocks) if len(blocks) > 1 else blocks[0]

    @staticmethod
    def _apply(parameter: Tensor, orthogonal: Tensor, group: dict[str, Any]) -> None:
        """The decoupled weight decay and the RMS-matched step of ``orthogonal`` on ``parameter``."""
        rate = rms_matched_rate(group["lr"], parameter.shape[0] // group["splits"], parameter.shape[1])
        parameter.mul_(1 - group["lr"] * group["weight_decay"])
        parameter.add_(orthogonal, alpha=-rate)

    def _step_sharded(self, matrices: list[tuple[DTensor, DTensor, dict[str, Any]]]) -> None:
        """The sharded matrices' updates (each a DTensor of this rank's rows) orthogonalised by their owners and their
        rows sent back (the module's docstring); every matrix shares one mesh, sharded on its rows."""
        mesh = matrices[0][0].device_mesh
        group, world = mesh.get_group(), mesh.size()
        rank = dist.get_rank(group)
        sent = []
        for index, (parameter, update, _) in enumerate(matrices):
            owner = index % world
            rows = _row_counts(parameter.shape[0], world)
            local = _padded(update.to_local().bfloat16(), max(rows))
            parts = [torch.empty_like(local) for _ in range(world)] if rank == owner else None
            work = _started(
                dist.gather(local, parts, dst=dist.get_global_rank(group, owner), group=group, async_op=True)
            )
            sent.append((owner, rows, parts, work))
        returned = []
        for (parameter, _, settings), (owner, rows, parts, work) in zip(matrices, sent, strict=True):
            work.wait()
            pieces = None
            if parts is not None:  # this rank's matrix: whole, orthogonalised, cut back into each rank's rows
                whole = torch.cat([part[:count] for part, count in zip(parts, rows, strict=True)])
                orthogonal = self._orthogonal(whole, settings)
                pieces = [_padded(piece, max(rows)) for piece in orthogonal.split(rows)]
            mine = torch.empty(max(rows), parameter.shape[1], dtype=torch.bfloat16, device=parameter.to_local().device)
            work = _started(
                dist.scatter(mine, pieces, src=dist.get_global_rank(group, owner), group=group, async_op=True)
            )
            returned.append((mine, rows, work))
        for (parameter, _, settings), (mine, rows, work) in zip(matrices, returned, strict=True):
            work.wait()
            local = mine[: rows[rank]]
            orthogonal = DTensor.from_local(
                local, parameter.device_mesh, parameter.placements, shape=parameter.shape, stride=parameter.stride()
            )
            self._apply(parameter, orthogonal, settings)


def _started(work: dist.Work | None) -> dist.Work:
    """An asynchronous collective's handle, which a rank of its group always gets.

    Raises:
        RuntimeError: If there is none: this rank is not in the mesh's group.
    """
    if work is None:
        raise RuntimeError("an asynchronous collective gave no handle: this rank is not in the mesh's group")
    return work


def _row_counts(rows: int, world: int) -> list[int]:
    """How many of a matrix's ``rows`` each of ``world`` ranks holds, sharded on its rows as FSDP2 shards them
    (``torch.chunk``, which may give fewer chunks than ranks: the last ranks then hold none)."""
    counts = [len(chunk) for chunk in torch.arange(rows).chunk(world)]
    return counts + [0] * (world - len(counts))


def _padded(rows: Tensor, count: int) -> Tensor:
    """``rows`` with zero rows after them up to ``count``: every rank sends and receives the same shape."""
    if rows.shape[0] == count:
        return rows.contiguous()
    return torch.cat([rows, rows.new_zeros(count - rows.shape[0], *rows.shape[1:])])
