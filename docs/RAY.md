# Running the page pipeline on Ray Data: what Ray does, what our stages do

Notes from a complete read of Ray's `doc/source/data` and `doc/source/core`: every page and API stub, the text and code
of the example notebooks, and the `doc_code` files the pages include. Where the docs are stubs or say nothing, they
are checked against Ray Data's source: `python/ray/data/{dataset,context}.py`, `_internal/compute.py`,
`_internal/remote_fn.py`, `_internal/execution/operators/actor_pool_map_operator.py`, `datasource/{datasink,
file_datasink}.py`, `_internal/datasource/webdataset_{datasink,datasource}.py`, `checkpoint/`,
`_internal/planner/planner.py` and `_internal/planner/checkpoint/plan_write_op.py`.
All of it is at ray master `ed1d83ac` (26 Oct 2026). Nothing here is tested on a cluster yet; the Ray work starts
after the tests on the Spark (TODO, "Inference at scale").

## Who does what

| Concern | Ray Data does it | Our stage code does it |
|---|---|---|
| Iterating rows, batching, scheduling, backpressure, pipelining CPU and GPU stages | yes | no: no loops, queues or schedulers of ours |
| A worker process that crashes (segfault, kernel OOM kill, node loss) | restarts the actor and retries its batch | nothing |
| An exception raised inside a stage | fails the block (all its rows); the run stops unless `max_errored_blocks` allows it | catch it per row and return the row with an error field |
| A GPU that can't be trusted any more (CUDA illegal access, a hung context) | replaces the actor only if its process dies | exit the process; Ray restarts the actor and retries the batch |
| Resuming a stopped run | job-level checkpointing (`CheckpointConfig`), only on a plan from a file read to a write | one row per page from read to sink, with a unique, stable id |
| Writing outputs | runs the sink's write tasks on the workers, retrying I/O errors (`retried_io_errors`) | a sink that names files by page id, so a rewrite replaces them |
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
  The error is a string: an exception holding an unpicklable field reaches the driver as a bare `RayError`.
- **GPU faults: exit, don't raise.** A raised `torch.AcceleratorError` is a user exception: not retried, and the actor
  stays alive on a broken CUDA context. Exiting the process (`os._exit(1)`) is a crash to Ray: the actor is recreated
  and the batch retried. Ray's own restart example does exactly this, and warns that "retried methods may execute
  twice, once on the failed actor and a second time on the restarted actor". `torch.OutOfMemoryError` usually leaves
  the context usable: retry it on the same actor through `actor_task_retry_on_errors=[torch.OutOfMemoryError]` with a
  smaller batch, or treat it like a fault. To verify on a local Ray before relying on it.
- **Constructors.** An actor that dies in `__init__` (a missing model, an OOM while loading) is not restarted by Ray
  Core; `max_consecutive_actor_init_deaths` (default 0: the first such death fails the run) and
  `actor_init_retry_on_errors` govern it. Set a small budget; check model paths before the run starts.
- **Out-of-memory kills.** The raylet's memory monitor kills workers above 95% of node memory, retriable ones first;
  killed tasks retry with backoff, killed actors restart (Ray Data's `max_restarts=-1`). Set `memory=` per stage to
  1.25 x the observed `max_uss_bytes` (`ray-data.log`) so Ray doesn't oversubscribe a node; enable cgroup v2 resource
  isolation, which reserves CPU and memory for Ray's own processes.
- **Idempotent work.** Retries and lineage reconstruction assume a task can run twice with the same result. Ours can:
  a crop's noise seed is fixed by its page id, so a retried row draws the same page.
- **The head node** holds the GCS; its loss ends the cluster unless the GCS is backed by HA Redis (KubeRay) or the
  alpha embedded RocksDB. Workers that can't reach it for `gcs_rpc_server_reconnect_timeout_s` (60 s) exit.
  Checkpointed runs restart from the last committed rows instead.

## Resuming: what checkpointing needs

- `DataContext.checkpoint_config = CheckpointConfig(id_column=..., checkpoint_path=...)`, with the path on S3 or a
  network filesystem every node mounts. Rerunning the same code skips committed rows ("each input row at least
  once"); `delete_checkpoint_on_success` defaults to True.
- **Only a plan from a file read to a write.** The planner checkpoints only when the plan ends in a write and every
  path starts at a file read (`_CHECKPOINT_FILTER_OPS = (Read, ReadFiles)`: `read_json`, `read_parquet`, ...).
  Otherwise it warns that "the logical plan doesn't support checkpointing" and runs without it, so a plan built with
  `from_items` is never resumable.
- **The id is unique and survives every stage.** "ID values must be unique across all rows in the dataset and must
  persist during all operators." Ray records the ids of the rows it wrote, block by block, and on a rerun drops the
  read rows whose id it recorded.
- **So: one row per page from read to sink.** A `flat_map` from a page to its crops gives several written rows the
  same id, and a page stopped after some of its crops would count as done and be skipped. Whole pages are one row
  already; window mode keeps a page's crops inside its row.
- **How the write commits.** Built-in file sinks (`write_parquet`, `write_json`, ...) get a two-phase commit: a
  pending checkpoint before the data file, committed after it; on restart, a crashed run's pending checkpoints and
  their data files are deleted. A custom `Datasink` is checkpointed after its write, so a retry may write the same
  rows again: its writes must be idempotent.

## Rows, batches and serialization

- `map` takes and returns a dict row with values of any type; `flat_map` returns a list of rows (not for a
  checkpointed page run, above); `map_batches` takes dicts of NumPy arrays (default), pandas or Arrow.
- Rows travel as Arrow blocks in the object store. Use native types: NumPy arrays, lists and dicts of them, strings
  and numbers. Images as `uint8` HxWx3 arrays (variable sizes arrive as an object array of arrays), not PIL objects;
  pydantic models cross a boundary as `model_dump(mode="json")` and are validated again on the other side. Torch
  tensors returned from a stage become NumPy; GPU tensors can't be serialized (`.cpu()` first). NumPy inputs are
  zero-copy and read-only ("assignment destination is read-only"): copy before writing into them.
- Ray can't split a row, so whole-page images make large rows: keep pages as compressed bytes or arrays per row and
  small batches.
- A GPU stage needs an explicit integer `batch_size` (rule of thumb: about a quarter of GPU memory); `map_batches`
  may hand over fewer rows than asked. Block order isn't preserved unless `preserve_order` is set: don't rely on it.

## Stages and resources

- A model stage is a class: `__init__` loads the model once per actor, `__call__` maps a batch (under
  `torch.inference_mode()`). Configuration reaches it through `fn_constructor_kwargs` (model paths, the judge's model
  and revision), not through import-time settings or `.env` files, which workers don't share.
- `compute=ActorPoolStrategy(size=n)` for a fixed pool or `(min_size, max_size)` to autoscale;
  `max_tasks_in_flight_per_actor` >= 2 overlaps input fetching with compute. `num_gpus` may be fractional (several
  actors per 96 GB card), but resources are logical: each actor must keep to its share of GPU memory itself.
- Ray Data counts an actor's CPUs as 0 unless the stage sets `num_cpus`, and its own CPU tasks need free CPU slots: if
  every slot is taken, "the experiment will appear to hang". Set `num_cpus` on GPU stages and leave headroom. Ray sets
  `OMP_NUM_THREADS` to a task's CPUs; cap MKL and OpenBLAS threads the same way.
- Separate CPU and GPU work into separate stages so Ray pipelines them: plan (CPU) -> draw (GPU) -> judge (GPU, its
  own pool) -> labels (CPU) -> sink. Our draw, judge and labels are already separate (`diffu_page.judge`,
  `blocks.ground_truth.crop_labels`).
- An actor's process lives as long as the pool: GPU memory hygiene (fragmentation, caches) is ours; Ray recycles
  workers only for GPU *tasks* (`max_calls=1`).
- Don't fork processes inside a stage (no DataLoader workers): Ray kills a worker's process group when it exits, but
  children hold GPU memory until then. Keep large objects out of closures (load them in `__init__`), and don't share
  state through globals: each worker is its own process.
- Turn off diffusers' progress bars in stages (`set_progress_bar_config(disable=True)`); per-step bars from hundreds of
  workers flood the logs.
- Model weights come baked into the image or from a shared model cache filled before the run, not downloaded by every
  actor at once.

## Inputs, outputs, packaging, monitoring

- Page jobs as JSON Lines or Parquet read with `read_json` / `read_parquet` (a file read, so the run can resume), on
  object storage or a shared filesystem mounted at the same path on every node. `local://` pins a read or write to the
  driver's node: for small tests only.
- Outputs through a sink, not files written inside a stage. The built-in `RowBasedFileDatasink` writes one file per
  row named by write task and row index, so it can't name a page's image, ALTO and PAGE files by its id. What fits:
  `write_webdataset` (alpha), which writes tar shards with one sample per row, a member per column named
  `{__key__}.{column}`: with `__key__` = page id and columns `png` (the uint8 array; its encoder makes the PNG),
  `alto.xml`, `page.xml` and `json`, a page is `{id}.png`, `{id}.alto.xml`, `{id}.page.xml` and `{id}.json`. It is a
  file sink, so it gets the two-phase commit, and WebDataset shards are a common format for VLM training. Otherwise
  `write_parquet` with one row per page (the same fields as columns), or our own `Datasink` writing loose files named
  by id (idempotent by name, checkpointed after the write). Drop large intermediate columns before the sink.
- The object store takes 30% of node memory in `/dev/shm` (a container needs a large `/dev/shm`) and spills to local
  disk once it is 80% full: point `--object-spilling-directory` at a fast disk with room, and watch
  `ray memory --stats-only`.
- Packaging: a container image for production; `runtime_env` (or `uv run` with the workspace's `pyproject.toml`)
  during development. A `working_dir` is at most 500 MiB, skips what `.gitignore` lists, must contain the editable
  packages, and its Ray and Python versions must match the cluster's.
- Secrets: `HF_TOKEN` from a Kubernetes secret, never in a `runtime_env` URI (Ray logs URIs and names directories
  after them). Token authentication (`RAY_AUTH_MODE=token`, the default for every cluster from Ray 2.61) set the same
  on every node.
- Watch `ray-data.log` (per-operator progress every 5 s), `Dataset.stats()` and the dashboard's metrics
  (`num_tasks_failed`, `num_restarting_actors`, GPU usage, rows out). A stage can add its own counters with
  `ray.util.metrics.Counter` (rows with an error, unclear lines) for a live label error rate.

## What we don't need, and why

- Compiled Graphs (beta) and Ray Direct Transport (alpha) cut Ray's ~1 ms call overhead and copy GPU tensors between
  actors without the object store; a page costs seconds of denoising and its image a few MB, so neither pays off.
- Placement groups reserve several GPUs together (multi-GPU training); our stages use one GPU or less each.
- Ray DAG, named and detached actors, namespaces, `ActorPool` and `ray.util.queue` are Core tools that Ray Data
  already replaces. Sandboxes (alpha) are for running untrusted code; ours isn't.

## What this means for our code

- `diffu_page.local_run.run_each` is the local commands' loop only. On Ray it isn't used: Ray Data iterates and
  retries; a stage handles one row's content errors itself.
- The runner's `DrawActor` catches every exception as a failed crop, CUDA faults included (TODO): split content
  errors (row error field) from device faults (exit the process).
- A page stays one row through every stage. `PageJob.id` is optional today; a Ray run requires it, unique and stable,
  checked when the jobs are read.
- Choose per run: `max_errored_blocks` (a small backstop), `actor_task_retry_on_errors` (transient errors only),
  `max_consecutive_actor_init_deaths` (a few), `checkpoint_config` (on), `memory=` and `num_cpus` per stage
  (measured), the spill directory and `/dev/shm` size.
