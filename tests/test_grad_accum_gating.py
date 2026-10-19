"""Regression test for the grad-accumulation gating fix (commits ac407f2 + f54ca40).

train.py's loop now gates ALL per-optimizer-step bookkeeping (EMA update, the `step`
counter, logging, save/val/sample cadence, the DDP barrier) behind

    if not accel.sync_gradients:
        continue

so under gradient_accumulation_steps>1 those run once per OPTIMIZER step, not once per
micro-batch. Pre-fix, `ema.step()` and `step += 1` ran every micro-batch, over-stepping
the EMA (= over-decayed / degraded export) and putting the step counter — and thus the LR
horizon and save/val/sample cadence — in the wrong units.

The real loop lives inline in train.train() (a monolithic main()); it is not importable as
a function, so this test FAITHFULLY REPRODUCES the exact gating construct from train.py
lines ~545-705 (the `with accel.accumulate(model)` block, the `if accel.sync_gradients`
clip, then `if not accel.sync_gradients: continue` gating `ema.step(params)` and
`step += 1`). It uses the REAL diffusers EMAModel (the same class train.py imports) and a
real accelerate Accelerator on CPU, so the observed `ema.optimization_step` / `step`
advance through the genuine accelerate accumulation machinery — not a hand-rolled mock of
sync_gradients.

Invariant asserted: across N micro-batches with gradient_accumulation_steps=G, both the
EMA step count (ema.optimization_step) and the gated `step` counter advance exactly N // G
times (once per optimizer step), NOT N times. Against the pre-fix code (ema.step + step+=1
every micro-batch) this would read N and FAIL.
"""

from __future__ import annotations

import torch
from accelerate import Accelerator
from diffusers.training_utils import EMAModel


def _run_gated_loop(grad_accum: int, num_micro_batches: int) -> tuple[int, int, int]:
    """Reproduce train.py's accumulate + sync_gradients gating on CPU.

    Returns (gated_step, ema_optimization_step, micro_batches_seen). `gated_step` mirrors
    train.py's `step` (incremented only past the `if not accel.sync_gradients: continue`
    guard); ema_optimization_step is the EMAModel's own optimizer-step counter.
    """
    accel = Accelerator(gradient_accumulation_steps=grad_accum, cpu=True)

    model = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model, opt = accel.prepare(model, opt)

    params = list(model.parameters())
    # Same construct as train.py:463 — EMAModel built from the prepared params.
    ema = EMAModel(params)

    step = 0  # train.py's `step` = optimizer steps once gated
    micro_seen = 0

    for _ in range(num_micro_batches):
        # ---- mirror train.py:545-562 (the non-stage2 branch) ----
        with accel.accumulate(model):
            out = model(torch.randn(2, 4))
            loss = out.pow(2).mean()
            accel.backward(loss)
            if accel.sync_gradients:  # train.py:558 — clip only on the real step
                accel.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
        micro_seen += 1

        # ---- mirror train.py:588-591, 705 (the gated bookkeeping) ----
        # The non-finite loss check runs inside the gated block at log steps (train.py) — omitted
        # here (loss is always finite for this dummy model); we focus on the gating itself.
        if not accel.sync_gradients:
            continue
        ema.step(params)  # train.py:590-591
        # (logging / save / val / sample / accel.wait_for_everyone() would run here too)
        step += 1  # train.py:705

    return step, ema.optimization_step, micro_seen


def test_gated_step_advances_once_per_optimizer_step():
    # 6 micro-batches, accumulate 2 -> 3 optimizer steps.
    step, ema_step, micro_seen = _run_gated_loop(grad_accum=2, num_micro_batches=6)
    assert micro_seen == 6  # every micro-batch ran the forward/backward
    # The fix: gated bookkeeping fires once per OPTIMIZER step (6 // 2 == 3), not per
    # micro-batch. Pre-fix this read 6 and would FAIL.
    assert step == 3
    assert ema_step == 3


def test_ema_not_over_stepped_under_grad_accum():
    # The original bug: EMA over-stepped (once per micro-batch) -> over-decayed shadow.
    # 8 micro-batches / accum 4 -> exactly 2 EMA steps, not 8.
    step, ema_step, _ = _run_gated_loop(grad_accum=4, num_micro_batches=8)
    assert ema_step == 2
    assert step == 2
    # Guard against a stale-mock pass: the pre-fix count (==num_micro_batches) must differ.
    assert ema_step != 8


def test_noop_at_grad_accum_one():
    # The fix is documented as a no-op at grad_accum=1 (grads sync every iteration), so the
    # gated counters must equal the micro-batch count there.
    step, ema_step, micro_seen = _run_gated_loop(grad_accum=1, num_micro_batches=5)
    assert micro_seen == 5
    assert step == 5
    assert ema_step == 5
