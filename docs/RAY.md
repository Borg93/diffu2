# Running the page pipeline on Ray Data: what Ray does, what our stages do

Notes from a full read of Ray's `doc/source/data` and the fault-tolerance, actor, task, object, serialization,
scheduling, out-of-memory, pattern and dependency pages of `doc/source/core`, checked against Ray Data's source where
the API pages are only stubs (`python/ray/data/{dataset,context}.py`, `_internal/compute.py`, `_internal/remote_fn.py`,
`_internal/execution/operators/actor_pool_map_operator.py`), all at ray master `ed1d83ac` (26 Oct 2026). Nothing here
is tested on a cluster yet; the Ray work starts after the tests on the Spark (TODO, "Inference at scale").

## Who does what

| Concern | Ray Data does it | Our stage code does it |
|---|---|---|
| Iterating rows, batching, scheduling, backpressure, pipelining CPU and GPU stages | yes | no: no loops, queues or schedulers of ours |
| A worker process that crashes (segfault, kernel OOM kill, node loss) | restarts the actor and retries its batch | nothing |
| An exception raised inside a stage | fails the block (all its rows); the run stops unless `max_errored_blocks` allows it | catch it per row and return the row with an error field |
| A GPU that can't be trusted any more (CUDA illegal access, a hung context) | replaces the actor only if its process dies | exit the process; Ray restarts the actor and retries the batch |
| Resuming a stopped run | job-level checkpointing (`CheckpointConfig`) on a read -> ... -> write pipeline | give every row a stable id |
| Validating a row | nothing (schemas are inferred) | pydantic at the stage boundary; judges for the image |

## Fault tolerance, as the source sets it

- **Actors are restarted by default.** Ray Data creates its actor pools with `max_restarts=-1` and
  `max_task_retries=-1` (`actor_pool_map_operator.py`, `_apply_default_remote_args`): a crashed actor is recreated by
  rerunning `__init__`, and the batch it was running is retried. Plain Ray Core defaults are 0 for both.
- **Only system errors are retried.** Actor tasks get `retry_exceptions=[RaySystemError]`
  (`remote_fn._add_system_error_to_retry_exceptions`), plus whatever `DataContext.actor_task_retry_on_errors` lists
  (default `False`). Map tasks (functions) retry user exceptions only with `retried_map_errors` and `max_map_retries`
  (default 0). A worker that dies is retried by Ray Core (`max_retries` 3 for tasks).
- **An exception in a stage fails the block.** "Ray Data drops the data in the failed blocks", and by default it
  tolerates no failed blocks (`max_errored_blocks=0`): one bad row would stop a run of millions. A small
  `max_errored_blocks` budget is a backstop, not the per-row mechanism.
- **Row-level tolerance lives inside the stage.** Ray's own LLM stages do it there (`should_continue_on_error=True`:
  "If a single row fails because of a request-level error from the engine, the job continues processing the remaining
  rows"). Ours: each stage catches a row's content error (an unreadable template, a judge that can't read a line) and
  returns the row with `error` set; later stages pass such rows through; the sink writes them to an errors table.
- **GPU faults: exit, don't raise.** A raised `torch.AcceleratorError` is a user exception: not retried, and the actor
  stays alive on a broken CUDA context. Exiting the process (`os._exit(1)`, the docs' own restart example) is a crash
  to Ray: the actor is recreated and the batch retried. `torch.OutOfMemoryError` usually leaves the context usable:
  retry it on the same actor through `actor_task_retry_on_errors=[torch.OutOfMemoryError]` with a smaller batch, or
  treat it like a fault. To verify on a local Ray before relying on it.
- **Constructors.** An actor that dies in `__init__` (a missing model, an OOM while loading) is not restarted by Ray
  Core; `max_consecutive_actor_init_deaths` (default 0: the first such death fails the run) and
  `actor_init_retry_on_errors` govern it. Set a small budget; check model paths before the run starts.
- **Out-of-memory kills.** The raylet's memory monitor kills workers above 95% of node memory; killed tasks retry
  with backoff, killed actors restart (Ray Data's `max_restarts=-1`). Set `memory=` per stage to 1.25 x the observed
  `max_uss_bytes` (`ray-data.log`) so Ray doesn't oversubscribe a node; enable cgroup v2 resource isolation.
- **Idempotent work.** Retries and lineage reconstruction assume a task can run twice with the same result. Ours can:
  a crop's noise seed is fixed by its page id, so a retried row draws the same page.
- **Resuming.** `DataContext.checkpoint_config = CheckpointConfig(id_column=..., checkpoint_path=<shared storage>)`
  on a pipeline that starts with a read and ends with a write; rerunning the same code skips committed rows ("each
  input row at least once"). Every page job needs a stable id (`PageJob.id`, a content hash), kept to the sink.
- **The head node** holds the GCS; its loss ends the cluster unless the GCS is backed by HA Redis (KubeRay) or the
  alpha embedded RocksDB. Checkpointed runs restart from the last committed rows instead.

## Rows, batches and serialization

- `map` takes and returns a dict row with values of any type; `flat_map` returns a list of rows (planning: one page
  in, its crops out); `map_batches` takes dicts of NumPy arrays (default), pandas or Arrow.
- Rows travel as Arrow blocks in the object store. Use native types: NumPy arrays, lists and dicts of them, strings
  and numbers. Images as `uint8` HxWx3 arrays (variable sizes arrive as an object array of arrays), not PIL objects;
  pydantic models cross a boundary as `model_dump(mode="json")` and are validated again on the other side. Torch
  tensors returned from a stage become NumPy; GPU tensors can't be serialized (`.cpu()` first). NumPy inputs are
  zero-copy and read-only: copy before writing into them.
- Ray can't split a row, so whole-page images make large rows: keep pages as compressed bytes or arrays per row and
  small batches.
- A GPU stage needs an explicit integer `batch_size` (rule of thumb: about a quarter of GPU memory); `map_batches`
  may hand over fewer rows than asked. Block order isn't preserved unless `preserve_order` is set: don't rely on it.

## Stages and resources

- A model stage is a class: `__init__` loads the model once per actor, `__call__` maps a batch. Configuration reaches
  it through `fn_constructor_kwargs` (model paths, the judge's model and revision), not through import-time settings
  or `.env` files, which workers don't share.
- `compute=ActorPoolStrategy(size=n)` for a fixed pool or `(min_size, max_size)` to autoscale;
  `max_tasks_in_flight_per_actor` >= 2 overlaps input fetching with compute. `num_gpus` may be fractional (several
  actors per 96 GB card), but resources are logical: each actor must keep to its share of GPU memory itself.
- Separate CPU and GPU work into separate stages so Ray pipelines them: plan (CPU) -> draw (GPU) -> judge (GPU, its
  own pool) -> labels (CPU) -> sink. Our draw, judge and labels are already separate (`diffu_page.judge`,
  `blocks.ground_truth.crop_labels`).
- An actor's process lives as long as the pool: GPU memory hygiene (fragmentation, caches) is ours; Ray recycles
  workers only for GPU *tasks* (`max_calls=1`).
- Don't fork processes inside a stage, keep large objects out of closures (load them in `__init__`), and don't share
  state through globals: each worker is its own process.

## Inputs, outputs, packaging, monitoring

- Page jobs as JSON Lines read with `ray.data.read_json`; files on object storage or a shared filesystem mounted at
  the same path on every node (no `local://`).
- Outputs through a sink, not files written inside a stage: `write_parquet`, or a `RowBasedFileDatasink` that writes
  one page image and its ALTO/PAGE XML per row.
- Packaging: a container image for production; `runtime_env` (or `uv run` with the workspace's `pyproject.toml`,
  editable packages included) during development.
- Watch `ray-data.log` (per-operator progress every 5 s), `Dataset.stats()` and the dashboard's metrics
  (`num_tasks_failed`, `num_restarting_actors`, GPU usage, rows out).

## What this means for our code

- `diffu_page.local_run.run_each` is the local commands' loop only. On Ray it isn't used: Ray Data iterates and
  retries; a stage handles one row's content errors itself.
- The runner's `DrawActor` catches every exception as a failed crop, CUDA faults included (TODO): split content
  errors (row error field) from device faults (exit the process).
- Choose per run: `max_errored_blocks` (a small backstop), `actor_task_retry_on_errors` (transient errors only),
  `max_consecutive_actor_init_deaths` (a few), `checkpoint_config` (on), `memory=` per stage (measured).
