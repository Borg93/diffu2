"""An ms-swift plugin (``swift sft --external_plugins <this file> --callbacks cer``): the ``cer`` callback, which reads
the sets :data:`~diffu_htr_train.evaluate.in_training.CER_SETS_ENV` names with the model being trained at each
evaluation and logs their error rates (see :mod:`diffu_htr_train.evaluate.in_training`). Imports ms-swift, so it
loads only in a training environment."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from swift.callbacks import TrainerCallback, callbacks_map
from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
from swift.utils import get_logger, unwrap_model_for_generation

from diffu_htr_train.evaluate.in_training import (
    PROMPT,
    CerSet,
    Reading,
    answer,
    at_start,
    cer_sets,
    metrics,
    write_readings,
)
from diffu_htr_train.evaluate.lines import EvalLine, read_lines

logger = get_logger()


def swift_read(engine: TransformersEngine, lines: Sequence[EvalLine], cer_set: CerSet) -> list[Reading]:
    """``lines`` read greedily by ``engine``, ``cer_set.batch`` at a time, each asked as the training samples ask."""
    engine.max_batch_size = cer_set.batch
    requests = [InferRequest(messages=[{"role": "user", "content": PROMPT}], images=[line.image]) for line in lines]
    responses = engine.infer(requests, RequestConfig(max_tokens=cer_set.max_tokens, temperature=0.0), use_tqdm=False)
    return [
        Reading(prediction=answer(response.choices[0].message.content), finished=response.choices[0].finish_reason)
        for response in responses
    ]


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _gathered(part: object, world: int) -> list[Any]:
    if world == 1:
        return [part]
    parts: list[Any] = [None] * world
    dist.all_gather_object(parts, part)
    return parts


class CerCallback(TrainerCallback):
    """Each rank reads its share of every set (every ``world``-th line), the shares gathered, the rates logged as
    ``eval_cer_<set>`` and the readings written to ``<output_dir>/cer/step-<n>.<set>.jsonl``. A set that fails to
    read is logged and skipped on every rank alike, so the run neither ends nor hangs on it."""

    def __init__(self, args: Any, trainer: Any) -> None:
        super().__init__(args, trainer)
        self.sets = {name: (cer_set, read_lines(cer_set.lines)) for name, cer_set in cer_sets().items()}
        self.at_start = at_start()
        logger.info(f"cer callback: {', '.join(f'{n} ({len(s[1])} lines)' for n, s in self.sets.items()) or 'no sets'}")

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.at_start:
            self._read_sets(args, state)

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._read_sets(args, state)

    def _read_sets(self, args: Any, state: Any) -> None:
        if not self.sets:
            return
        trainer = self.trainer
        rank, world = _rank_world()
        was_training = trainer.model.training
        logs: dict[str, float] = {}
        with (
            unwrap_model_for_generation(
                trainer.model_wrapped, trainer.accelerator, gather_deepspeed3_params=args.ds3_gather_for_generation
            ) as model,
            trainer.template.generate_context(),
            torch.no_grad(),
        ):
            model.eval()
            engine = TransformersEngine(model, template=trainer.template)
            for name, (cer_set, lines) in self.sets.items():
                started = time.perf_counter()
                mine = list(range(rank, len(lines), world))
                try:
                    part: tuple[list[int], list[Reading] | None, str | None] = (
                        mine,
                        swift_read(engine, [lines[k] for k in mine], cer_set),
                        None,
                    )
                except Exception as error:  # noqa: BLE001 - a monitor's failure must not end a run of days
                    part = (mine, None, repr(error))
                parts = _gathered(part, world)
                failures = [failure for _, _, failure in parts if failure]
                if failures:
                    logger.warning(f"cer callback: {name} not read at step {state.global_step}: {failures[0]}")
                    continue
                readings: list[Reading] = [Reading(prediction="", finished=None)] * len(lines)
                for indices, got, _ in parts:
                    for k, reading in zip(indices, got, strict=True):
                        readings[k] = reading
                logs |= {f"eval_{key}": value for key, value in metrics(name, lines, readings).items()}
                logs[f"eval_cer_seconds_{name}"] = round(time.perf_counter() - started, 1)
                if rank == 0:
                    write_readings(
                        Path(args.output_dir) / "cer" / f"step-{state.global_step}.{name}.jsonl", lines, readings
                    )
        if was_training:
            trainer.model.train()
        if logs:
            trainer.log(logs)


callbacks_map["cer"] = CerCallback
