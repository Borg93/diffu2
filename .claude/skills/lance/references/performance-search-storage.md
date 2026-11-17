# Lance: performance, full-text search, tokenizers, object store

Source: Lance docs at 15.0.0-beta, `docs/src/guide/`. Installed: pylance 13.0.0 (`.venv`).
Names marked **[not in 13.0.0]** were checked against the installed package (Python signatures and
strings in the compiled library) and are absent. See the last section.

What these four pages do NOT cover: `cleanup_old_versions`, auto cleanup, the cost of many versions,
compaction thresholds (`target_rows_per_fragment` etc.), NFS, file locking on local disks. Look in
the pages on data evolution / table maintenance / transactions for those.

---

## docs/src/guide/performance.md

### Logging and tracing (Python)

- pylance logs to stderr by default. Environment variables:
  - `LANCE_LOG`: level and target filter (env_logger syntax); replaces `RUST_LOG`.
  - `LANCE_TRACING`: tracing level, default `info`; `debug` gives more spans for performance work.
  - `LANCE_LOG_STYLE`: `auto` | `always` | `never`.
  - `LANCE_LOG_TS_PRECISION`: `ns` | `us` | `ms` | `s`.
  - `LANCE_LOG_FILE`: write logs to this path (parent directories created; falls back to stderr).
- Trace events appear in pylance logs under the prefix `lance::events::`, e.g.
  `LANCE_LOG="warn,lance::events::object_store::throttle=info"`.
- Event targets and fields:
  - `lance::file_audit`: `mode` (create, delete, delete_unverified), `type` (manifest, data file,
    index file, deletion file).
  - `lance::dataset_events`: `event` (loading, writing, committed, deleting, ...), `uri`, `mode`,
    `operation`, `predicate`, `columns`.
  - `lance::io_events`: `type` (open_scalar_index, open_vector_index, load_vector_part,
    load_scalar_part). Not emitted when the index comes from the in-memory cache, so they show
    cache misses.
  - `lance::execution`: `type` (only `plan_run`), `output_rows`, `iops`, `bytes_read`,
    `indices_loaded`, `parts_loaded`, `index_comparisons`, `peak_reserved_bytes`.
  - `lance::object_store::throttle`: cloud stores only.
  - `lance::memory`: Linux only, on with `LANCE_MEMORY_OBSERVE=1`; covers index builds
    (`create_index`, `merge_index_metadata`), fields `event`, `pid`, `operation_id`, `operation`,
    `index_type`, `stage`, `peak_bytes`, `rss_bytes`, `cgroup_bytes`, `cgroup_limit_bytes`.
    A start with no completion = the build failed, was cancelled, or the process died (e.g. OOM).
    `LANCE_MEMORY_OBSERVE=1 LANCE_LOG="warn,lance::events::memory=info"` logs only these.

### Threads

- Lance APIs are thread-safe and may be called concurrently unless stated otherwise. Operations may
  run in parallel on one table; some conflict (see conflict handling below).
- Two pools:
  - I/O pool: local stores 8 threads by default, cloud stores 64. Override: `LANCE_IO_THREADS`.
    If raised, raise `io_buffer_size` too.
  - Compute pool: number of cores. Override: `LANCE_CPU_THREADS`. Set it when several Lance
    processes share a machine. Decoding is compute-heavy, so a scan that looks I/O-bound still
    needs compute threads.

### Caches

- Metadata cache: LRU sized in bytes, default 1 GiB; holds manifests, transactions, deletion files,
  row id structures, index metadata, file metadata. Keys are scoped to the dataset URI.
- Index cache: LRU sized in bytes, default 6 GiB; set with `index_cache_size_bytes` when opening
  (`lance.dataset(uri, index_cache_size_bytes=...)`). `index_cache_size` (entries) is deprecated
  since 0.30.0.
- Neither cache is shared between tables by default. Rule: open a dataset once and share the
  object across the application, or create one session and pass it when opening tables
  (`lance.dataset(..., session=...)`).
- `dataset.session().size_bytes()`: deep-size estimate of the session.
- `dataset.session().cache_diagnostics()`, `cache_diagnostics(refresh=True)`,
  `cache_diagnostics(by_type=True)`; `session.set_index_cache_capacity(...)`,
  `session.set_metadata_cache_capacity(...)` **[not in 13.0.0]**. Returned dict:
  `diagnostics["index"]`, `diagnostics["metadata"]`, each with `["activity"]["hits"]`,
  `["activity"]["misses"]`, `["backend"]["size_bytes"]`, `["backend"]["capacity_bytes"]`,
  `["utilization"]`.
- Reading diagnostics: a cold run raises `misses`, `loads_started`, `loads_succeeded`; a warm repeat
  should raise only `hits`. Growing `size_removals` with repeated loads = cache too small. A full
  cache alone means little; compare hit/miss deltas around the same operation.

### Scan memory

- Scans stream, but memory is set by `io_buffer_size` and `batch_size`.
- Pages are typically 8 to 32 MB: allow about 32 MB per I/O thread.
- `io_buffer_size` default 2 GB (64 pages). It is a soft limit (cannot read less than one page).
- `batch_size` default 8192 rows. Each CPU thread holds one decoded batch.
- Aim for batches around 1 MB for scalar data. For wide columns turn `batch_size` down:
  1024-dim float32 x 8192 rows = 32 MB per batch; x 16 CPU threads = 512 MB per scan. The page
  suggests 1024 rows per batch for such columns.
- Upper bound: `(2 * io_buffer_size) + (batch_size * num_compute_threads)` bytes (batch size in
  bytes here).
- Batches handed to the application are no longer tracked: do not collect them all (`to_table`) if
  memory matters; iterate `scanner.to_batches()`.
- Lance reads whole encoded pages: a smaller `batch_size` shrinks decoded batches, not the first
  range request (one encoded page per selected column).
- Scanner controls:
  - `fragment_readahead`: fragments with reads scheduled at once. `1` = one fragment at a time.
  - `batch_readahead`: concurrent batch decoding; does not change storage request sizes.
  - `io_buffer_size`: buffered I/O bytes; applies backpressure when decoding falls behind.
  - `scan_in_order=True`: orders the returned batches only; fragment reads still overlap.
  - `LANCE_IO_THREADS`: concurrent storage requests for the whole process.
- Conservative settings for a slow or shared connection (the page writes this for remote stores;
  applying it to NFS is our inference):

```python
# LANCE_IO_THREADS=8 python scan.py
scanner = dataset.scanner(
    fragment_readahead=1,
    batch_readahead=2,
    io_buffer_size=64 * 1024 * 1024,
)
for batch in scanner.to_batches():
    process(batch)
```

- Transparent huge pages for big scans (long strings, JSON, vectors): opt in with
  `GLIBC_TUNABLES=glibc.malloc.hugetlb=1 python scan.py` when
  `/sys/kernel/mm/transparent_hugepage/enabled` is `madvise`. Measured on an 11 GB string scan:
  9x fewer page faults, scan 2 to 3.5x faster, full rewrite to 2.2 about 1.5x faster. Takes of a
  few rows unaffected. Pitfall: with `/sys/kernel/mm/transparent_hugepage/defrag` = `madvise` and
  fragmented memory, one 256 MiB allocation went from 0.09 s to 1.2 s. Measure on the host first.

### Cloud throttling (not relevant to local disks)

- S3/GCS/Azure are wrapped in an AIMD rate limiter; local and in-memory stores are not throttled.
- Options `lance_aimd_initial_rate` (2000), `lance_aimd_min_rate` (1), `lance_aimd_max_rate`
  (10000), `lance_aimd_decrease_factor` (0.5), `lance_aimd_additive_increment` (300),
  `lance_aimd_burst_capacity` (100); env vars are the upper-case names with `LANCE_` prefix.
  Replaces the old `LANCE_PROCESS_IO_THREADS_LIMIT`.

### Fragment sizing

- Manifest-level cost grows with the NUMBER of fragments: every mutation (append, metadata update,
  schema change, compaction) rewrites the manifest; opening a dataset, listing fragments, planning
  a scan and dataset-level conflict resolution all walk it.
- Fragment-level cost grows with the SIZE of a fragment: scans of a matching fragment, compaction,
  updates, deletes, `merge_insert`. Conflict detection for these is per fragment.
- Numbers:
  - Default 1M rows per fragment (`max_rows_per_file=1048576` in `lance.write_dataset`); good up to
    about 1B rows. Beyond that, about 100M rows per fragment is reasonable.
  - "Tens of thousands of fragments per table is generally fine."
  - 10 GB to 100 GB per fragment is a reasonable upper range; 1 TB is a hard ceiling (S3 object
    cap is 5 TB).
  - Many concurrent updates, deletes or `merge_insert`: prefer more fragments (fewer retries).
- For us: 3.2 M rows appended 2,000 at a time = about 1,600 fragments, inside the "generally fine"
  range on fragment count. The page gives no rule on the number of versions and no small-fragment
  threshold that forces compaction.

### Conflicts and compaction

- Optimistic concurrency. On conflict one operation is retried automatically; the retry repeats
  work already done.
- Common conflicts: (1) compaction running with index building (both modify the same indices);
  (2) updates that touch the same fragments (both rewrite the same data files).
- Compaction is among the most expensive writes: it rewrites data files and, by default, remaps
  every index to the new row addresses. Concurrent with index building it tends to fail and retry;
  repeated failure degrades the table layout.
- Fragment Reuse Index: compaction records old-to-new row address mapping instead of remapping
  indices; the mapping is applied when an index is loaded into the cache (small load cost, no
  query cost once cached). Compaction and index building then no longer conflict.

```python
dataset.optimize.compact_files(defer_index_remap=True)
```

  (`defer_index_remap` is present in 13.0.0's `compact_files` signature.)

### Index costs

- BTree: sorted copy of the column plus 4 bytes per value for the row id (as stated) plus a lookup
  structure of about 0.001 % of the column. Trained with an external sort, spills to disk, low
  memory. Updating merges new sorted values in linear time (no full re-sort). Loaded into the
  index cache in pages of 4096 values. Sort is O(n log n).
- Bitmap: one Roaring bitmap per distinct value. Training needs at least 8 bytes of memory per row
  plus the hash map keys; many distinct values at scale is memory-heavy. Fast for equality and
  very small ranges; large ranges are "extremely slow", use BTree for those.
- `parts_loaded` in `lance::execution` = pages (BTree) or bitmaps (Bitmap) read from disk.
- Vector index:
  - IVF training sample in memory: `num_partitions * sample_rate * dimension * sizeof(data_type)`;
    default `sample_rate` 256. Example: 1024 partitions x 256 x 768 x 4 bytes = 768 MiB. If the
    dataset has fewer rows than `num_partitions * sample_rate`, all rows are used.
  - Shuffle: streams the whole column; 2 GiB I/O readahead (`LANCE_DEFAULT_IO_BUFFER_SIZE`),
    batches of 8,192 rows, `num_cpus - 2` batches in flight (`LANCE_CPU_THREADS`). Memory about
    `io_readahead_buffer + num_cpu_threads * batch_size * (raw_vector_size + transformed_vector_size)`.
    About 8 MiB writer buffer per partition.
  - On disk per row: PQ `num_sub_vectors + 8` bytes; SQ `dimension + 8` bytes; RQ (RaBitQ) 1-bit
    `dimension / 8 + 20`, multi-bit
    `dimension / 8 + round_up(dimension, 64) * (num_bits - 1) / 8 + 28`; new RQ indexes default to
    5 bits per dimension; `num_bits=1` minimises size at lower recall.
  - Centroids: `num_partitions * dimension * sizeof(data_type)` (10K x 768 float32 = 30 MiB).
  - AMX: automatic on Linux x86_64 with AMX-FP16 for `float16` vectors with `dot` distance,
    `dimension >= 32`, `num_centroids >= 32`; disable with `LANCE_DISABLE_AMX=1`. Changes partition
    assignment (exact instead of approximate), so indexes built with and without differ.

---

## docs/src/guide/full_text_search.md

- Engine: inverted index, BM25 scoring, result column `_score` (higher is better). Supports token
  match, boolean, exact phrase, fuzzy, multi-column.
- Substring filters (`contains(...)`) are a different thing: use an N-gram or FM scalar index; no
  BM25 ranking there.

### Create

```python
dataset.create_scalar_index("body", "INVERTED")
dataset.create_scalar_index("body", "INVERTED", with_position=True)   # needed for phrase queries
```

- Default index name: `<column>_idx`. Pass `name=...` when one column needs two indexes (e.g.
  inverted and N-gram).
- `with_position=True`: larger index, no effect on non-phrase query speed.
- Analyzer options (keyword arguments to `create_scalar_index`):
  - `base_tokenizer`: `simple`, `whitespace`, `raw`, `icu`, or language-specific tokenizers.
  - `lower_case`, `stem`, `remove_stop_words`, `ascii_folding`.
  - `language`: language for stemming and default stop words.
  - `max_token_length`: exclusive limit in UTF-8 bytes; tokens at or over it are dropped.
  - `analyzer="code"`: code profile **[`analyzer` not seen in 13.0.0's `create_scalar_index`]**.
- Large builds: `num_workers` and `memory_limit` control parallelism and memory (both appear in
  13.0.0's `create_scalar_index` source).

### Query (Python)

```python
from lance.query import (BooleanQuery, CombinedFieldsQuery, FullTextOperator,
                         MatchQuery, MultiMatchQuery, Occur, PhraseQuery)

dataset.to_table(full_text_query="train Boston")            # string: all indexed text columns
dataset.to_table(full_text_query=MatchQuery("train Boston", "body", operator=FullTextOperator.AND))
dataset.to_table(full_text_query=MatchQuery("ramen broth", "body", operator=FullTextOperator.OR))
dataset.to_table(full_text_query=MatchQuery("train", "body"), filter="category = 'travel'")
dataset.to_table(full_text_query=PhraseQuery("ramen broth", "body"))
dataset.to_table(full_text_query=BooleanQuery([
    (Occur.MUST, MatchQuery("train", "body")),
    (Occur.MUST_NOT, MatchQuery("Boston", "body")),
]))
dataset.to_table(full_text_query=MatchQuery("ramen", "body", fuzziness=1, max_expansions=50))
dataset.to_table(full_text_query=MultiMatchQuery("columnar guide", ["title", "body"], boosts=[2.0, 1.0]))
dataset.to_table(full_text_query=CombinedFieldsQuery("Lance multimodal", ["title", "body"],
                                                     operator=FullTextOperator.AND))
```

- `MatchQuery`: Levenshtein fuzzy matching with `fuzziness`; `fuzziness=None` = automatic by term
  length; `prefix_length` = exact prefix required before edits; `max_expansions` (50 in examples).
  Useful for historical spelling variants.
- `MultiMatchQuery`: best score from any column wins; `boosts` per column.
- `CombinedFieldsQuery` (BM25F): terms may be spread across columns, corpus statistics combined.
  The columns must use the same tokenizer and index configuration.
- Phrase queries require an index built with `with_position=True`. Stop words are absent from
  indexed positions: if phrases must contain stop words, build with `remove_stop_words=False`.
- 13.0.0 signatures (verified): `MatchQuery(query, column, *, boost=1.0, fuzziness=0,
  max_expansions=50, operator=FullTextOperator.OR, prefix_length=0, document_granularity=None)`;
  `PhraseQuery(query, column, *, slop=0, document_granularity=None)`;
  `MultiMatchQuery(query, columns, *, boosts=None, operator=OR)`;
  `CombinedFieldsQuery(query, columns, *, boosts=None, operator=OR)`;
  `BooleanQuery(queries: list[tuple[Occur, FullTextQuery]])`.

### Maintain

- Appended rows stay searchable: indexed search is combined with a scan of unindexed rows. That
  scan grows with unindexed data, so fold new fragments in periodically:

```python
dataset.insert(new_rows)
dataset.optimize.optimize_indices(index_names=["body_idx"])
stats = dataset.stats.index_stats("body_idx")
print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
```

- `optimize_indices()` adds new data to existing indexes; it does NOT change analyzer settings.
- After changing tokenizer or analyzer settings: `create_scalar_index(..., replace=True)` to
  rebuild over the whole dataset.
- No numeric threshold is given for when to run `optimize_indices`.
- Full-text and vector search can be combined in one scanner (see `vector_search.md`).

---

## docs/src/guide/tokenizer.md

- Built-in: ICU, Jieba, Lindera. ICU has bundled segmenter data (no download). Jieba (Chinese) and
  Lindera (Japanese/Korean: `ipadic`, `ko-dic`, `unidic`) need downloaded models. No Swedish-specific
  tokenizer is on this page; Swedish goes through `language` + `stem` (see full_text_search.md) with
  `simple` or `icu`.
- Model directory: env `LANCE_LANGUAGE_MODEL_HOME`, default
  `${system data directory}/lance/language_models`. Own models go in that directory.

### Inspect tokens without an index

```python
import lance
tokens = lance.tokenize("the Cats and Dogs")
[(token.text, token.position) for token in tokens]
# [("cat", 0), ("dog", 2)]
```

- `lance.tokenize` exists in 13.0.0. It accepts the tokenizer options of
  `LanceDataset.create_scalar_index` (custom stop words, n-grams, code analyzer).
- Positions start at the first kept token and keep the gaps left by stop words and filters (same
  representation as phrase matching).
- Options set to `None` take the analyzer profile's default. Exception: omitting
  `max_token_length` keeps the limit of 40 UTF-8 bytes; `max_token_length=None` disables it.
- Use this to test historical Swedish spellings before building an index: check what stemming,
  `ascii_folding` (å/ä/ö would fold to a/a/o) and stop-word removal do to real transcription lines.

### Token length limit

- `max_token_length` is exclusive and in UTF-8 BYTES, default 40. `"x" * 39` kept, `"x" * 40`
  dropped; `"é" * 20` (40 bytes) dropped. To keep a 40-byte token set at least 41, or `None`.
- Pitfall for Swedish: å, ä, ö are 2 bytes each, so long compounds reach the limit sooner than
  their character count suggests.

### ICU

```python
ds.create_scalar_index("text", "INVERTED", base_tokenizer="icu")
ds.create_scalar_index("text", "INVERTED", base_tokenizer="icu/split")
```

- Unicode word-boundary rules; good for mixed-language text. `icu/split` also splits
  punctuation-delimited identifiers into searchable parts.

### Code analyzer (not for our text)

- `analyzer="code"` or `base_tokenizer="code"`; requires FTS format v3. Flags:
  `split_identifiers` (False), `split_on_numerics` (True), `preserve_original` (True),
  `index_operators` (False). Disables stemming and stop-word removal. Existing code indices must be
  rebuilt to get the new length normalisation.

### Jieba / Lindera

- `python -m lance.download jieba`; model in `${LANCE_LANGUAGE_MODEL_HOME}/jieba/default`;
  `base_tokenizer="jieba/default"`; user dictionaries via `config.json` (`"main"`, `"users"`).
- `python -m lance.download lindera -l [ipadic|ko-dic|unidic]` (needs `lindera-cli` to compile);
  `base_tokenizer="lindera/ipadic"`; `config.yml` in the model root or `LINDERA_CONFIG_PATH`.

---

## docs/src/guide/object_store.md

### How configuration is passed

- The store is chosen by the URI scheme of the dataset path. A plain path is the local filesystem.
- Options: environment variables (global) or `storage_options={...}` (per dataset) on
  `lance.dataset(...)` and `lance.write_dataset(...)`. Values are strings.

### Local filesystem

- `block_size`: gap in bytes under which two reads of one file are merged into one request.
  Default `4096` for local files, `65536` for object stores. Larger = fewer requests for scattered
  reads (e.g. a `take` of rows spread across a file) at the price of reading the bytes between.
  The page gives no NFS advice; raising it for a table on NFS is our inference, to be measured.
- Durability, `local_fsync` **[not in 13.0.0: the string is absent from the compiled library]**:

```python
lance.write_dataset(
    pa.table({"id": [1]}),
    "/path/to/data.lance",
    storage_options={"local_fsync": "true"},
)
```

  - Default `"false"`: by default Lance does NOT fsync local writes before returning.
  - When `"true"`: data, transaction and manifest files are synced before a write reports success
    (also files copied in deep clones); on Unix also their directory entries, including new
    directories. Raises write latency.
  - It is a store setting, not a dataset property: pass it again every time the dataset is reopened
    for writing. Only `"true"` and `"false"` are accepted.
- Bulk copies and deep clones on local paths ignore `LANCE_IO_SERVER_SIDE_COPY_ENABLED`.
- `scheduler_error_mode`: `best_effort` (default) or `fail_fast` (first error cancels the other
  physical reads of the same logical request) **[string absent from 13.0.0's library]**.

### NFS

- The page does not mention NFS at all: no statement on atomic rename, locking, close-to-open
  caching, or whether concurrent writers are safe on a network filesystem. Treat a table on NFS
  with one writer at a time until tested or confirmed from the transaction/commit documentation.

### Commit safety and concurrent writers

- Nothing on this page states how commits are made safe on a local filesystem. The stated rules are
  for object stores:
  - `s3://`: safe concurrent writes need the store to enforce create-only `PutObject` with
    `If-None-Match: *` atomically. Some S3-compatible stores (Garage is named) accept the header and
    ignore it: two writers both report success and acknowledged data is silently lost. Lance does
    not check for this.
  - Tencent COS: a custom commit lock is required for writes.
  - `goosefs://`: uses `ConditionalPutCommitHandler`; older Lance releases use `UnsafeCommitHandler`
    there, so mixed-version writers are not safe.
- Reads never need a commit lock (stated for S3 and COS).
- Custom commit lock (Python): pass `commit_lock` to `lance.write_dataset` when creating or
  appending, and to `lance.dataset` when opening for later writes. The callback takes the manifest
  version to commit and returns a context manager that takes the lock on entry and releases it on
  exit, also when the commit raises. Every writer of the dataset must use the same lock service and
  key; a process-local lock is not enough across processes. (`commit_lock` is in 13.0.0's
  signatures of both functions.)
- General lesson stated for GooseFS and applicable as caution: a commit guarantee holds only when
  every writer uses the same commit handler, so do not mix Lance versions among writers of one
  table; quiesce writers before upgrading.

### Other general options

- `timeout` (default `30s`, whole request), `connect_timeout` (`5s`), `download_retry_count` (`3`),
  `client_max_retries` (`3`), `client_retry_timeout` (`180` s), `allow_http`,
  `allow_invalid_certificates`, `user_agent`, `proxy_url`, `proxy_ca_certificate`, `proxy_excludes`,
  `object_store_cache_prune_threshold` (default `64`, `0` disables). These concern HTTP stores.
- Per-base options: `base_<id>.<key>` applies to one registered base path; ids from
  `ds.base_paths()`; `base_store_params` (keyed by base URI) takes precedence.
- Deep clone: four concurrent non-local files by default; `LANCE_DEEP_CLONE_STREAM_CONCURRENCY`.

### Cloud stores (three lines)

- S3 `s3://` (`region`, `endpoint`, `access_key_id`, `secret_access_key`, `session_token`,
  `aws_provider_scheme` = `token` | `ecs` | `irsa`, `s3_express`); S3-compatible stores need both
  `region` and `endpoint`; Apache Ozone goes through the S3 path.
- GCS `gs://` (`service_account`), Azure `az://` (`account_name`, `account_key`), OSS `oss://`,
  TOS `tos://`, COS `cos://`, GooseFS `goosefs://`.
- Hugging Face `hf://datasets/<owner>/<repo>/<path>` (`hf_token`, `hf_revision`,
  `hf_download_mode` = `http` | `xet`, `hf_enable_resolve_cache`).

---

## Unclear or possibly newer than 13.0.0

Checked against the installed pylance 13.0.0 (Python signatures, and strings in the compiled
library; a missing string is strong but not final evidence):

- Absent in 13.0.0:
  - `local_fsync` storage option. So on 13.0.0 there is no documented way to make local writes
    durable before they return; a power loss right after a commit can lose it.
  - `session.cache_diagnostics(...)`, `session.set_index_cache_capacity(...)`,
    `session.set_metadata_cache_capacity(...)`. 13.0.0's session has only
    `index_cache_size_bytes`, `is_same_as`, `size_bytes`.
  - `scheduler_error_mode`, `object_store_cache_prune_threshold`.
  - `LANCE_MEMORY_OBSERVE` and the `lance::memory` events; `peak_reserved_bytes` in
    `lance::execution`.
  - `analyzer="code"`, `split_identifiers` in `create_scalar_index` (not in its Python source; the
    arguments pass through `**kwargs`, so not fully excluded).
- Present in 13.0.0: `lance.tokenize`, `lance.download`, all five query classes,
  `compact_files(defer_index_remap=...)`, `optimize_indices(**kwargs)`, scanner arguments
  `fragment_readahead`, `batch_readahead`, `io_buffer_size`, `batch_size`, `scan_in_order`;
  `commit_lock`, `session=`, `index_cache_size_bytes`, `metadata_cache_size_bytes`,
  `base_store_params`, `ds.base_paths()`; env strings `LANCE_IO_THREADS`, `LANCE_CPU_THREADS`,
  `LANCE_DEFAULT_IO_BUFFER_SIZE`, `LANCE_LOG_FILE`, `LANCE_DISABLE_AMX`,
  `LANCE_LANGUAGE_MODEL_HOME`, `LANCE_IO_SERVER_SIDE_COPY_ENABLED`,
  `LANCE_DEEP_CLONE_STREAM_CONCURRENCY`, `lance_aimd_initial_rate`.
- Not verified either way: `fuzziness=None` (13.0.0 types it `int = 0`); the default RaBitQ
  5 bits per dimension; huge-page timings (measured on 2.0 and 2.2 files); whether
  `dataset.optimize.optimize_indices(index_names=[...])` accepts `index_names` in 13.0.0 (it takes
  `**kwargs`); the ICU tokenizer names `icu` and `icu/split` (the string `icu` appears in
  `create_scalar_index`'s source, not tested by building an index).
- Unclear in the docs themselves:
  - BTree "additional 4 bytes per value ... to store the row ID" while vector indexes state an
    8-byte row id.
  - `(2 * io_buffer_size) + (batch_size * num_compute_threads)` mixes bytes and rows; read
    `batch_size` as bytes per batch.
  - Nothing on NFS, on how local commits are made atomic, or on several processes writing one
    local table.
  - Nothing on the number of versions, on `cleanup_old_versions`, or on a small-fragment threshold
    for compaction; only `compact_files(defer_index_remap=True)` and the fragment sizing numbers.
  - Which `language` values exist (Swedish stemming and stop words are not listed on these pages).
