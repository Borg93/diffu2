# Running the page pipeline on Ray Data: what Ray does, what our stages do

Notes from a complete read of Ray's `doc/source/data` and `doc/source/core`: every page and API stub, the text and code
of the example notebooks, and the `doc_code` files the pages include. Where the docs are stubs or say nothing, they
are checked against Ray Data's source: `python/ray/data/{dataset,context}.py`, `_internal/compute.py`,
`_internal/remote_fn.py`, `_internal/execution/operators/actor_pool_map_operator.py`, `datasource/{datasink,
file_datasink}.py`, `_internal/datasource/webdataset_{datasink,datasource}.py`, `checkpoint/`,
`_internal/planner/planner.py` and `_internal/planner/checkpoint/plan_write_op.py`.
All of it is at ray master `ed1d83ac` (26 Oct 2026). Our sink is Lance: the Lance parts come from lance-ray's docs
and sources (`datasink.py`, `io.py`, `fragment.py`, `datasource.py`; 0.5.0 at `b4c1a2e`) and every page of Lance's
guide (`docs/src/guide` at `51cf89d`). Nothing here is tested on a cluster yet; the Ray work starts after the tests
on the Spark (TODO, "Inference at scale").

## Who does what

| Concern | Ray Data does it | Our stage code does it |
|---|---|---|
| Iterating rows, batching, scheduling, backpressure, pipelining CPU and GPU stages | yes | no: no loops, queues or schedulers of ours |
| A worker process that crashes (segfault, kernel OOM kill, node loss) | restarts the actor and retries its batch | nothing |
| An exception raised inside a stage | fails the block (all its rows); the run stops unless `max_errored_blocks` allows it | catch it per row and return the row with an error field |
| A GPU that can't be trusted any more (CUDA illegal access, a hung context) | replaces the actor only if its process dies | exit the process; Ray restarts the actor and retries the batch |
| Resuming a stopped run | job-level checkpointing (`CheckpointConfig`), but not safely with a Lance sink (below) | skip the page ids the Lance table already holds; a unique, stable id per page |
| Writing outputs | runs the sink's write tasks; lance-ray's sink writes fragments in them and commits once on the driver | one row per page in a Lance table, keyed by the page id |
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
  A rerun resumes from the pages already committed to the Lance table.

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
- **Not with Lance.** The Lance sinks (lance-ray's and Ray's own) are plain `Datasink`s: each write task writes
  uncommitted fragment files, and the driver commits them all as one transaction (`on_write_complete`) after the
  last task. Ray writes a non-file sink's checkpoint in each write task right after its write, before that commit.
  If the run dies before the commit, the checkpoint lists pages that never reached the table, and a rerun skips
  them. Read in both sources (26 Oct); to reproduce on a local Ray when the Ray work starts. Our runs resume from
  the table instead (Lance section).

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

- Page jobs as JSON Lines (`read_json`) or a Lance table (`lance_ray.read_lance`), on object storage or a shared
  filesystem mounted at the same path on every node. `local://` pins a read or write to the driver's node: for small
  tests only.
- Outputs through a sink, not files written inside a stage: a Lance table written with lance-ray (Lance section).
  Drop large intermediate columns before the sink.
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

## Lance: the page table

- **Writing.** `lance_ray.write_lance(ds, uri, mode="create" | "append" | "overwrite")` uses `LanceDatasink`: each
  write task writes fragment files (retrying `LanceError(IO)` up to 10 times), and the driver commits all of them as
  one new version at the end. Nothing is visible before that commit; files of failed attempts stay until
  `cleanup_old_versions` removes unreferenced files (after 7 days by default).
- **Set `min_rows_per_file`.** Its default, 1,048,576 rows, is also the minimum number of rows Ray hands each write
  task, and the sink holds all of a task's blocks in memory for retries. For pages of about 1 MB, set it to a few
  hundred.
- **Resume from the table.** A run commits once, so size runs so that losing one is acceptable. Before a run, read
  the committed `id` column (a BTREE index on `id` keeps that cheap), drop those jobs and write with
  `mode="append"`; `merge_insert("id").when_not_matched_insert_all()` inserts only rows whose id is new, so a rerun
  can't duplicate a page. lance-ray's `stream=True` commits after every batch instead, but every byte passes through
  the driver, and its `resume_rows` skips rows by position, which Ray Data keeps stable only with `preserve_order`.
- **Later passes add columns** instead of rewriting pages: `lance_ray.add_columns` / `add_columns_from` run a function
  over the table on Ray (a VLM judge's verdicts, embeddings for QC) and commit the new columns without copying the
  images.
- **Upkeep.** Many appends leave many small fragments: `lance_ray.compact_files`. Tag released versions
  (`ds.tags.create`); tagged versions survive `cleanup_old_versions`.
- **Storage.** S3-compatible stores need `region` and `endpoint` in `storage_options`; concurrent writers need a store
  that enforces conditional `PutObject` (`If-None-Match: *`) or a shared commit lock.

The page schema, written with file format `data_storage_version="2.2"` (Blob v2 needs 2.2 or later; 2.3 is unstable):

| Column | Arrow type | Why |
|---|---|---|
| `id` | `utf8` | the page id: the resume and `merge_insert` key; BTREE index |
| `image` | Blob v2 (`blob_field` / `blob_array`), the encoded page | a whole page is about 0.9 MB as JPEG and several MB as PNG; a blob column keeps it out of compaction rewrites, and `read_blobs` reads it for training |
| `alto_xml`, `page_xml` | `large_utf8` | the labels; no dots in column names (filters can't address them) |
| `text` | `large_utf8` | what was written; an INVERTED (full-text) index if we want to search it |
| `width`, `height` | `int32` | |
| `seed` | `int64` | redraws the page |
| `template`, `style`, `model`, `lora` | `utf8` | provenance |
| `lines` | `list<struct<read: utf8, cer: float32, numbers_differ: bool>>` | the judge's verdict per line (`LineVerdict`) |
| `unclear_lines` | `int32` | |
| `error` | `utf8`, nullable | a row that failed, with why |

Blob v2 places a value by size: up to 64 KiB inline in the data file, 64 KiB to 4 MiB in shared `.blob` sidecar files,
larger in its own `.blob` file (thresholds per column). Plain `binary` suits small values read whole (thumbnails, crops
of a few tens of KB). lance-ray's reader turns blob columns back into bytes (through `take_blobs`), so a Ray reader
gets `large_binary`. `pa.json_()` columns (stored as JSONB) can be filtered with `json_get_*` if the provenance
grows into nested metadata.

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
- The runner writes Parquet (`diffu_runner/run.py`, `write_parquet`); our sink is Lance: `lance_ray.write_lance` with
  the page schema above (TODO).
- Choose per run: `max_errored_blocks` (a small backstop), `actor_task_retry_on_errors` (transient errors only),
  `max_consecutive_actor_init_deaths` (a few), `memory=` and `num_cpus` per stage (measured), `min_rows_per_file`,
  the spill directory and `/dev/shm` size. `checkpoint_config` stays off with a Lance sink.
