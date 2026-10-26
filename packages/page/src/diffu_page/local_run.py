"""The local commands' loop over many items (diffu-generate, diffu-evaluate, the training caches, on one machine): one
item that fails is recorded by key with its error and the rest still run, unless it failed in a way that makes every
later item fail too (a device that can no longer be trusted).

Not for a cluster: there Ray Data iterates the rows and retries crashed workers, a stage turns a failing row into a row
with its error, and a stage exits its worker on a device fault so Ray restarts it and retries the batch (docs/RAY.md).
Nothing here schedules, retries or queues."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from diffu_page.output import err_console

NOT_RUN = "not run: an earlier item failed on the device"


def describe(error: BaseException) -> str:
    """An error as a run records it: its type and message."""
    return f"{type(error).__name__}: {error}"


def run_each[I, K, T](
    items: Sequence[I],
    fn: Callable[[I], T],
    *,
    key: Callable[[I], K],
    stop_on: tuple[type[BaseException], ...] = (),
) -> tuple[list[T], dict[K, str]]:
    """``fn`` over ``items``: the results of those that ran, and each failure's error by ``key``. An error of a type in
    ``stop_on`` ends the run: every item after it is recorded as not run."""
    done: list[T] = []
    failed: dict[K, str] = {}
    for position, item in enumerate(items):
        try:
            done.append(fn(item))
        except stop_on as error:
            failed[key(item)] = describe(error)
            failed |= dict.fromkeys((key(later) for later in items[position + 1 :]), NOT_RUN)
            err_console.print(f"{key(item)} failed on the device, the run stops: {failed[key(item)]}")
            break
        except Exception as error:  # one item must not end a long run; the caller's record keeps the error
            failed[key(item)] = describe(error)
            err_console.print(f"{key(item)} failed: {failed[key(item)]}")
    return done, failed
