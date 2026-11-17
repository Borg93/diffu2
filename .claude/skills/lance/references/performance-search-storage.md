Covers: scan and index-build memory, threads, caches, fragment sizing, compaction clashes, full-text search and tokenizers for Swedish text, local-disk write safety, logging.
Read when: tuning a scan or an index build, building or querying an INVERTED index, choosing a fragment size, or judging what a local write guarantees.

Each section names its page of the Lance docs (15.0.0-beta, `docs/src/guide/`). **[absent 13.0.0]**: the name is missing from the installed pylance (Python signatures and strings of the compiled library; a missing string is strong but not final evidence). **[unverified 13.0.0]**: not checked either way.

## Scan within a memory budget

(guide/performance.md)

- Memory is set by `io_buffer_size` (default 2 GB = 64 pages; a soft limit, one page is the minimum read) and `batch_size` (default 8192 rows; each CPU thread holds one decoded batch). Pages are typically 8 to 32 MB: allow about 32 MB per I/O thread.
- Upper bound: `(2 * io_buffer_size) + (batch_size * num_compute_threads)`. The docs mix rows and bytes here: read `batch_size` as bytes per batch.
- Aim for batches near 1 MB for scalar data. 1024-dim float32 x 8192 rows = 32 MB a batch, x 16 CPU threads = 512 MB a scan: hence 1024 rows for wide columns.
- Iterate `scanner.to_batches()` when memory matters: batches handed to the application are no longer tracked, so `to_table` holds them all.
- A smaller `batch_size` shrinks decoded batches only; the first range request stays one whole encoded page per selected column.
- Scanner arguments: `fragment_readahead` (fragments with reads scheduled at once; `1` = one at a time), `batch_readahead` (concurrent batch decoding; storage request sizes unchanged), `io_buffer_size` (backpressure when decoding falls behind), `scan_in_order=True` (orders returned batches only; fragment reads still overlap).
- Slow or shared connection (written for remote stores; applying it to NFS is our inference):

```python
# LANCE_IO_THREADS=8 python scan.py
scanner = dataset.scanner(fragment_readahead=1, batch_readahead=2, io_buffer_size=64 * 1024 * 1024)
for batch in scanner.to_batches(): ...
```

- Big scans of long strings, JSON or vectors: opt in to transparent huge pages with `GLIBC_TUNABLES=glibc.malloc.hugetlb=1 python scan.py` when `/sys/kernel/mm/transparent_hugepage/enabled` is `madvise`. Measured on an 11 GB string scan (2.0 and 2.2 files, [unverified 13.0.0]): 9x fewer page faults, scan 2 to 3.5x faster, full rewrite to 2.2 about 1.5x faster; takes of a few rows unaffected. Pitfall: with `.../transparent_hugepage/defrag` = `madvise` and fragmented memory, one 256 MiB allocation went from 0.09 s to 1.2 s. Measure on the host first.

## Set threads and caches

(guide/performance.md)

- I/O pool: 8 threads for local stores; `LANCE_IO_THREADS` sets it for the whole process; raise `io_buffer_size` with it.
- Compute pool: number of cores; `LANCE_CPU_THREADS`. Set it when several Lance processes share a machine. Decoding is compute-heavy, so a scan that looks I/O-bound still needs compute threads.
- Open a dataset once and share the object, or pass one session to every open (`lance.dataset(..., session=...)`): the caches are per table otherwise.
- Metadata cache: LRU, default 1 GiB (manifests, transactions, deletion files, row id structures, index and file metadata), keyed by dataset URI. Index cache: LRU, default 6 GiB, `lance.dataset(uri, index_cache_size_bytes=...)`; `index_cache_size` (entries) is deprecated since 0.30.0.
- `dataset.session().size_bytes()` gives a deep-size estimate. 13.0.0's session has only `index_cache_size_bytes`, `is_same_as`, `size_bytes`.
- **[absent 13.0.0]** `session.cache_diagnostics(refresh=, by_type=)`, `session.set_index_cache_capacity`, `session.set_metadata_cache_capacity`. Reading diagnostics: compare hit/miss deltas around one operation (a warm repeat should raise only `hits`); `size_removals` growing with repeated loads = cache too small; a full cache alone means little.

## Choose a fragment size

(guide/performance.md)

- Cost per NUMBER of fragments: every mutation (append, metadata update, schema change, compaction) rewrites the manifest; open, fragment listing, scan planning and dataset-level conflict resolution walk it.
- Cost per SIZE of a fragment: scans of a matching fragment, compaction, updates, deletes, `merge_insert`; their conflict detection is per fragment, so many concurrent ones favour more fragments (fewer retries).
- Default 1M rows a fragment (`max_rows_per_file=1048576` in `lance.write_dataset`), good to about 1B rows; beyond, about 100M rows a fragment. "Tens of thousands of fragments per table is generally fine." 10 to 100 GB a fragment is a reasonable upper range; 1 TB a hard ceiling.
- Ours: 3.2 M rows appended 2,000 at a time = about 1,600 fragments, inside "generally fine".

## Compact beside index builds and updates

(guide/performance.md)

- Concurrency is optimistic: on a conflict one operation retries automatically and repeats its work. Common clashes: compaction with index building (same indices); updates touching the same fragments (same data files).
- Compaction rewrites data files and by default remaps every index to the new row addresses; clashing with an index build it fails and retries, and repeated failure degrades the table layout.
- `dataset.optimize.compact_files(defer_index_remap=True)` writes a Fragment Reuse Index instead: the old-to-new address mapping is applied when an index loads into the cache (small load cost, no query cost once cached), and compaction no longer clashes with index building.

## Budget an index build

(guide/performance.md)

- BTREE: sorted copy of the column + 4 bytes a value for the row id (as stated; vector indexes state an 8-byte row id) + a lookup structure of about 0.001 % of the column. External sort that spills to disk: low memory, O(n log n). An update merges new sorted values in linear time. Cached in pages of 4096 values.
- BITMAP: one Roaring bitmap per distinct value. Training needs at least 8 bytes of memory a row plus the hash-map keys: many distinct values at scale is memory-heavy. Use it for equality and very small ranges; large ranges are "extremely slow", give those a BTREE.
- Vector, IVF training sample in memory: `num_partitions * sample_rate * dimension * sizeof(data_type)`, `sample_rate` default 256 (1024 x 256 x 768 x 4 bytes = 768 MiB); with fewer rows than `num_partitions * sample_rate`, all rows are used.
- Vector, shuffle: streams the whole column with 2 GiB readahead (`LANCE_DEFAULT_IO_BUFFER_SIZE`), batches of 8,192 rows, `num_cpus - 2` batches in flight (`LANCE_CPU_THREADS`). Memory about `io_readahead_buffer + num_cpu_threads * batch_size * (raw_vector_size + transformed_vector_size)`, plus about 8 MiB writer buffer a partition.
- Vector, on disk a row: PQ `num_sub_vectors + 8` bytes; SQ `dimension + 8`; RQ 1-bit `dimension / 8 + 20`; RQ multi-bit `dimension / 8 + round_up(dimension, 64) * (num_bits - 1) / 8 + 28`. New RQ indexes default to 5 bits a dimension [unverified 13.0.0]; `num_bits=1` is smallest at lower recall. Centroids: `num_partitions * dimension * sizeof(data_type)` (10K x 768 float32 = 30 MiB).
- AMX is automatic on Linux x86_64 with AMX-FP16 for `float16` vectors with `dot` distance, `dimension >= 32`, `num_centroids >= 32`. It makes partition assignment exact, so indexes built with and without it differ: set `LANCE_DISABLE_AMX=1` when two builds must match.

## Build a full-text index on Swedish text

(guide/full_text_search.md, guide/tokenizer.md)

- `dataset.create_scalar_index("body", "INVERTED", with_position=True)`. Phrase queries need `with_position=True` (larger index, other queries as fast). Stop words are absent from indexed positions: build with `remove_stop_words=False` when phrases must contain them.
- Default index name `<column>_idx`; pass `name=...` when one column carries two indexes (inverted and N-gram). Substring filters (`contains(...)`) use an N-gram or FM scalar index, with no BM25 ranking.
- Analyzer keyword arguments: `base_tokenizer` (`simple`, `whitespace`, `raw`, `icu`, language-specific), `lower_case`, `stem`, `remove_stop_words`, `ascii_folding`, `language` (stemming and default stop words), `max_token_length`. Large builds: `num_workers`, `memory_limit` (both in 13.0.0's source).
- Swedish has no tokenizer of its own on these pages: use `language` + `stem` with `simple` or `icu`. Which `language` values exist is not listed.
- `base_tokenizer="icu"`: Unicode word boundaries, data bundled, good for mixed-language text; `"icu/split"` also splits punctuation-delimited identifiers. The string `icu` is in 13.0.0's `create_scalar_index` source; no index was built to test it.
- `max_token_length` is exclusive and in UTF-8 BYTES, default 40: `"x" * 39` kept, `"x" * 40` dropped, `"é" * 20` dropped. å, ä, ö are 2 bytes each, so long Swedish compounds reach the limit early: set 41 or more, or `None` to disable. Omitting the argument keeps 40; other options set to `None` take the analyzer profile's default.
- Test real transcription lines first with `lance.tokenize(text, **the same options)` (present in 13.0.0): `[(t.text, t.position) for t in lance.tokenize("the Cats and Dogs")]` gives `[("cat", 0), ("dog", 2)]`. Positions keep the gaps left by stop words and filters, as phrase matching sees them. Check what stemming, stop-word removal and `ascii_folding` (å/ä/ö would fold to a/a/o) do to historical spellings.
- Leave aside: `analyzer="code"` (needs FTS format v3; `analyzer` and `split_identifiers` are not in 13.0.0's Python source, though `**kwargs` pass through) and Jieba/Lindera (Chinese, Japanese/Korean; downloaded models).

## Query full text

(guide/full_text_search.md)

Scoring is BM25 in column `_score`, higher is better. Full-text and vector search combine in one scanner.

```python
dataset.to_table(full_text_query="train Boston")   # a string searches all indexed text columns
dataset.to_table(full_text_query=MatchQuery("train", "body"), filter="category = 'travel'")
dataset.to_table(full_text_query=BooleanQuery([(Occur.MUST, MatchQuery("train", "body")),
                                               (Occur.MUST_NOT, MatchQuery("Boston", "body"))]))
```

13.0.0 signatures (verified), all in `lance.query` with `FullTextOperator` and `Occur`:

- `MatchQuery(query, column, *, boost=1.0, fuzziness=0, max_expansions=50, operator=FullTextOperator.OR, prefix_length=0, document_granularity=None)`. `fuzziness` is Levenshtein distance, the tool for historical spelling variants; `prefix_length` = exact prefix required before edits. `fuzziness=None` = automatic by term length in the docs [unverified 13.0.0: typed `int = 0`].
- `PhraseQuery(query, column, *, slop=0, document_granularity=None)`.
- `MultiMatchQuery(query, columns, *, boosts=None, operator=OR)`: the best score from any column wins.
- `CombinedFieldsQuery(query, columns, *, boosts=None, operator=OR)`: BM25F, terms may spread across columns; the columns must share tokenizer and index configuration.
- `BooleanQuery(queries: list[tuple[Occur, FullTextQuery]])`.

## Keep a text index current

(guide/full_text_search.md)

- Appended rows stay searchable through a scan of unindexed rows that grows with them. Fold them in with `dataset.optimize.optimize_indices(index_names=["body_idx"])` (`index_names` [unverified 13.0.0]: the method takes `**kwargs`). The docs give no threshold: watch `dataset.stats.index_stats("body_idx")["num_unindexed_rows"]` against `["num_indexed_rows"]`.
- `optimize_indices` keeps the analyzer settings. After changing tokenizer or analyzer: `create_scalar_index(..., replace=True)` rebuilds over the whole dataset.

## Write to the local disk safely

(guide/object_store.md)

- A plain path is the local filesystem. Store options go in `storage_options={...}` (string values) on `lance.dataset` and `lance.write_dataset`, or in environment variables.
- Durability: Lance does not fsync local writes before returning. `storage_options={"local_fsync": "true"}` **[absent 13.0.0]** syncs data, transaction and manifest files and their directory entries before success, at higher write latency; it is a store setting, so pass it at every open for writing. On 13.0.0 no documented way makes a local write durable: a power loss right after a commit can lose it.
- Commit safety: the page states nothing on how local commits are made atomic, on several processes writing one local table, or on NFS (atomic rename, locking, close-to-open caching). Keep a table on NFS to one writer at a time until tested.
- Use one Lance version among all writers of a table and quiesce writers before upgrading: a commit guarantee holds only when every writer uses the same commit handler (stated for GooseFS, a caution elsewhere).
- `commit_lock` (in 13.0.0 on `lance.write_dataset` and `lance.dataset`): a callback taking the manifest version and returning a context manager that locks on entry and releases on exit, also when the commit raises. Every writer needs the same lock service and key; a process-local lock does not span processes. Reads never need it.
- `block_size`: gap in bytes under which two reads of one file merge into one request; default `4096` local, `65536` object stores. Larger = fewer requests for scattered reads (a `take` across a file) at the price of the bytes between. Raising it on NFS is our inference, to be measured.
- **[absent 13.0.0]** `scheduler_error_mode` (`best_effort` default | `fail_fast`), `object_store_cache_prune_threshold`.
- Cloud, in three lines. The URI scheme picks the store (`s3://`, `gs://`, `az://`, `hf://datasets/...`, others); S3-compatible stores need both `region` and `endpoint`.
  Concurrent `s3://` writes are safe only where create-only `PutObject` with `If-None-Match: *` is enforced atomically: some stores (Garage is named) ignore the header and acknowledged data is silently lost; Lance does not check.
  Cloud stores alone are throttled (AIMD, `lance_aimd_*` options) and use 64 I/O threads.

## Trace what Lance did

(guide/performance.md)

- pylance logs to stderr. `LANCE_LOG` (level and target filter, env_logger syntax; replaces `RUST_LOG`), `LANCE_TRACING` (default `info`; `debug` adds spans), `LANCE_LOG_STYLE`, `LANCE_LOG_TS_PRECISION`, `LANCE_LOG_FILE` (falls back to stderr).
- Events sit under the prefix `lance::events::`, e.g. `LANCE_LOG="warn,lance::events::memory=info"`:
  - `lance::file_audit`: files created and deleted, by `mode` and `type`.
  - `lance::dataset_events`: `event` (loading, writing, committed, deleting, ...) with `uri`, `operation`, `predicate`, `columns`.
  - `lance::io_events`: `type` (open_scalar_index, open_vector_index, load_vector_part, load_scalar_part). Emitted only on an index-cache miss, so they count misses.
  - `lance::execution`: `type` (only `plan_run`), `output_rows`, `iops`, `bytes_read`, `indices_loaded`, `parts_loaded` (BTREE pages or bitmaps read from disk), `index_comparisons`, `peak_reserved_bytes` **[absent 13.0.0]**.
  - `lance::memory` **[absent 13.0.0]**: Linux, `LANCE_MEMORY_OBSERVE=1`, index builds; fields include `stage`, `peak_bytes`, `rss_bytes`, `cgroup_limit_bytes`. A start with no completion = the build failed, was cancelled, or the process died (e.g. OOM).

## Silent in these pages

`cleanup_old_versions`, auto cleanup, the cost of many versions, a small-fragment threshold or other compaction thresholds (`target_rows_per_fragment` etc.), file locking on local disks: look in the pages on data evolution, table maintenance and transactions.
