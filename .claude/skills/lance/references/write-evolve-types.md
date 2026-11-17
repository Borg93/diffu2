# Lance: writing, evolving and typing a table (Python)

Source: the Lance guide pages at docs version 15.0.0-beta, each read in full. Installed: pylance 13.0.0.
Everything below is what the page states. Where a name was checked against the installed 13.0.0
package (signature inspection only, nothing was run against data), the last section says so.

Docs root: `<scratchpad>/lance_docs/docs/src/guide/`

---

## docs/src/guide/read_and_write.md

### Create and append
- `lance.write_dataset(data, uri)` accepts `pyarrow.Table`, `pandas.DataFrame`, `pyarrow.dataset.Dataset`, `Iterator[pyarrow.RecordBatch]`.
- Streaming an iterator requires `schema=`: `lance.write_dataset(producer(), uri, schema=schema, mode="overwrite")`.
- Append without rewriting existing fragments: `ds.insert(new_table)` or `ds = lance.write_dataset(new_table, ds, mode="append")`.

### Versions and handles
- The format is immutable; every mutation commits a new dataset version.
- In-place methods (`delete`, etc.) advance that handle to the committed version. Other open handles keep reading their pinned snapshot until reopened or checked out to latest.

### Delete
- `dataset.delete("name = 'Bob'")` (SQL filter). Rows are marked in a deletion file, not rewritten; indices are not invalidated.

### Update by SQL
- `dataset.update({"name": "'Bob'"}, where="name = 'Blob'")`. Values are SQL expressions: string literals need inner single quotes; expressions may reference columns (`dataset.update({"age": "age + 2"})`, no `where` = all rows).
- Pitfall: a Python loop of `dataset.update(..., where=f"name='{name}'")` per row "works, but is inefficient"; use merge insert for a set of individual rows.

### Merge insert (key join)
- `dataset.merge_insert("name").when_matched_update_all().execute(new_table)` — bulk update of existing rows.
- `.when_not_matched_insert_all()` — insert if not exists. Both together = upsert.
- `.when_not_matched_insert_all().when_not_matched_by_source_delete("age >= 40")` — replace a filtered region.
- Uses a JOIN internally: "you typically want this column to be a unique key or id".
- Pitfall: modified rows "will be removed and inserted back into the table, changing their position to the end"; relative order may change (hash join). Same is said of the update operation.

### Read
- `lance.dataset(uri)`; `ds.to_table(columns=[...], filter="...", limit=, offset=)`; `ds.to_batches(...)` takes the same parameters; `ds.take([1, 100, 500], columns=[...])`.
- Filter expressions supported: `>`, `>=`, `<`, `<=`, `=`, `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`, `IS TRUE`/`IS NOT TRUE`/`IS FALSE`/`IS NOT FALSE`, `IN`, `LIKE`, `NOT LIKE`, `regexp_match(column, pattern)`, `CAST`, `any_element(list_column, 'predicate')` (experimental).
- `IN` list syntax in the example uses brackets: `label IN [10, 20]`.
- Nested access by subscript: struct by field name (`note['email']`), list by index.
- Column names with special characters or SQL keywords: backticks; for nested fields each path segment is backticked.
- Limit: "Field names containing periods (`.`) are not supported."
- Typed literals: `date '2021-01-01'`, `timestamp '2021-01-01 00:00:00'`, `decimal(8,3) '1.000'`; `timestamp(0|3|6|9)` = s/ms/us/ns, default 6.
- SQL→Arrow: `float`→`Float32`, `double`→`Float64`, `int`→`Int32`, `bigint`→`Int64`, `string`→`Utf8`, `binary`→`Binary`, `boolean`→`Boolean`, `date`→`Date32`.
- `any_element(list_col, 'pred')` / `array_any_match(col, d -> ...)` on `list<struct>`: experimental, syntax may change. A `listelement` scalar index (`IndexConfig(index_type="listelement", parameters={"keys": [...]})`) answers it; also experimental.

### Compaction
- `dataset.optimize.compact_files(target_rows_per_fragment=1024 * 1024)`: fragments already above the count are skipped; others merged in fragment-id order, so data order is preserved.
- Creates a new version; does not delete old versions or their files.
- Can remove deleted rows; rewritten fragments have no deletion files.
- Pitfall: compaction changes `_rowaddr` of the compacted rows. Run `compact_files()` before `optimize_indices()` or index rebuilds. For identifiers that survive, enable stable row IDs.

### Cleanup
- Storage grows with every insert, update and delete, because old versions keep referencing old files until removed. "an update rewrites the affected columns for the affected rows."
- `dataset.cleanup_old_versions()` — default removes versions older than two weeks. `older_than=timedelta(days=1)` overrides.
- Never deletes the current version; `older_than=timedelta(0)` is safe and removes all but the current one.
- Removed versions cannot be recovered or time-travelled to. `lance.dataset(uri, asof=t)` returns the latest *retained* version before `t` (may be older than the one current at `t`); `ValueError` only when none predates `t`. `version=N` fails if N was removed.
- Tagged versions, current versions and branch dependencies are protected.
- Unverified files (e.g. left by aborted or failed commits) are removed only if at least 7 days old.
- `delete_unverified=True` drops that age protection. Danger, as stated: use only when "no other process is operating on the dataset at all"; a concurrent write may have its files deleted, "leaving the dataset corrupted".
- `delete_concurrency` (default: object store I/O parallelism, `LANCE_IO_THREADS`), `delete_rate_limit` (per second); positive integers.
- Multi-window retention: `from lance import CleanupPolicy, KeepOne, RemoveAll`; `dataset.cleanup_old_versions(policy=CleanupPolicy(rules=[KeepOne(per_interval=..., older_than=...), ..., RemoveAll(older_than=...)]))`. Ages must strictly increase; `RemoveAll` optional and last; `KeepOne.older_than` at least two days; do not combine `policy` with `older_than` or `keep_one`. `explain_cleanup_old_versions` takes the same `policy` for a read-only preview. `versions` and `retain_versions` limit what can be removed.
- Auto cleanup: `lance.write_dataset(..., auto_cleanup_options=AutoCleanupConfig(interval=20, older_than_seconds=3600))`; on an existing dataset `ds.optimize.enable_auto_cleanup(AutoCleanupConfig(...))`, `ds.optimize.disable_auto_cleanup()`; or `ds.update_config({"lance.auto_cleanup.interval": "20", "lance.auto_cleanup.older_than": "3600s", ...})`.
- Auto cleanup runs in the commit path; `skip_auto_cleanup=True` on `write_dataset` skips it for one write.

### Data file version
- `data_storage_version` selects the format of newly written files. The dataset property is the default for writes that omit a target; create and overwrite set it; append, update, merge-insert and compaction do not change it. V1 and V2 cannot be mixed.
- Per-operation override: `ds.update({...}, data_storage_version="2.2")`, `ds.merge_insert("id")....data_storage_version("2.2").execute(...)`, `ds.optimize.compact_files(data_storage_version="2.2")`.
- `"2.3"` is unstable: files of one unstable revision may be unreadable by a later one; experimentation only. `"stable"` / `"next"` resolve per engine release; use exact versions for reproducible output.
- Before writing files that differ from the dataset default, upgrade every reader and writer to a mixed-version-aware release. The commit sets paired reader/writer feature flags automatically; they stay set; older clients reject such snapshots.

### Concurrency stated on this page
- Old versions give readers snapshot isolation from concurrent writers.
- `delete_unverified=True` is unsafe with any concurrent writer (above).

---

## docs/src/guide/data_evolution.md

### Concurrency (the page's one rule)
- "schema changes will conflict with most other concurrent write operations." If a schema change and an append run together, one of them fails, depending on order. "it's recommended to perform schema changes when no other writes are happening."

### Add columns, schema only (all NULL)
- `dataset.add_columns(pa.field("embedding", pa.list_(pa.float32(), 128)))` or `dataset.add_columns(pa.schema([("label", pa.string()), ("score", pa.float32())]))`.
- Metadata-only, "very fast". Stated purpose: add the column, then populate it lazily with a large distributed job.
- File format `<= 2.1`: cannot add sub-columns under an existing `struct`. From `2.2`: can extend nested struct fields, including structs inside lists.

### Add columns with backfill
- SQL: `dataset.add_columns({"hash": "sha256(name)", "status": "'active'"})` — expressions may reference existing columns; a literal sets one value for all rows.
- UDF: function takes a `pyarrow.RecordBatch`, returns a `pyarrow.RecordBatch` or pandas DataFrame, called once per batch:
  ```python
  @lance.batch_udf(checkpoint_file="embedding_checkpoint.sqlite")
  def add_random_vector(batch):
      embeddings = np.random.rand(batch.num_rows, 128).astype("float32")
      return pa.RecordBatch.from_arrays(
          [pa.FixedSizeListArray.from_arrays(embeddings.flatten(), 128)],
          names=["embedding"])
  dataset.add_columns(add_random_vector)
  ```
- `checkpoint_file` lets a failed UDF restart from the last checkpoint. Pitfall: the file "can get quite large, since it needs to store unsaved results for up to an entire data file".

### Add pre-computed columns: merge
- `dataset.merge(new_data, "id")` — give a table holding the join column and the new columns, and the name of the join column. Fills columns "without having to rewrite the whole dataset".
- The page shows only the case where the new table has every key. It does not say what unmatched rows get, nor what happens with duplicate keys or with a column name that already exists.

### Drop columns
- `dataset.drop_columns(["name"])` — metadata-only; data stays on disk.
- To reclaim space: `dataset.optimize.compact_files(...)` then `cleanup_old_versions()`.
- Reversible only while old versions are retained; after compaction plus cleanup the data may be unrecoverable. Advice: tag before the drop, delay cleanup.
- From format `2.2`: nested sub-column removal (e.g. `people.item.city`) is supported.

### Rename
- `dataset.alter_columns({"path": "id", "name": "new_id"})`; nested: `{"path": "meta.id", "name": "new_id"}`.

### Cast
- `dataset.alter_columns({"path": "embedding", "data_type": pa.list_(pa.float16(), 128)})`.
- Rewrites that column to new data files; other columns are not rewritten.
- Pitfall: "If the column has an index, the index will be dropped if the column type is changed."
- The page's example is exactly float32 → float16 vectors "to save disk space at the cost of lower precision".

---

## docs/src/guide/update_join.md

### What it is
- Options on `LanceFragment.update_columns`, a "low-level, fragment-oriented API". It returns new fragment metadata and the ids of the modified fields; "the caller must commit those results to the dataset". The commit workflow is in `distributed_write.md#update-columns` (not one of the pages read here).

### Call
```python
updated_fragment, fields_modified = fragment.update_columns(
    updates,
    left_on="id",
    right_on="id",
    strategy="auto",                 # "auto" | "hash" | "sort_merge"
    max_hash_rows=250_000,
    max_hash_bytes=GIB,
    external_memory_pool_bytes=256 * MIB,
    max_temp_directory_bytes=20 * GIB,
)
```
- `max_hash_rows` and `max_hash_bytes`: both omitted or both given.

### Semantics and constraints
- Rows with a matching right-hand key get the supplied values; rows without a match keep their values.
- The right-hand side must hold the join column and at least one *existing* fragment column. (So this call does not add a new column.)
- Join-key types must match on both sides.
- `right_on` is removed from the payload and cannot be updated. If `left_on` differs from `right_on`, a replacement column named `left_on` is allowed and updates the left key.
- `_rowid` and `_rowaddr` cannot be replacement columns.
- Empty right-hand side is a no-op.
- Pitfall: duplicate right-hand keys — "Lance selects one matching RHS row, but which row is chosen is not defined."

### Strategies
- `"auto"` (default): hash join while the right-hand side is within both thresholds (250,000 rows and 1 GiB estimated allocation); otherwise sort-merge. The byte figure is an estimate (Arrow buffers, keys, hash-table overhead), not on-disk size and not a hard limit.
- `"hash"`: always in memory; the thresholds, pool and temp limit do not constrain it; a large input can exhaust memory.
- `"sort_merge"`: spillable, bounded by `external_memory_pool_bytes`; payload columns are late-materialized.
- Defaults: pool = operation setting, else `LANCE_MEM_POOL_SIZE`, else 256 MiB per execution partition (one partition today). Temp limit = operation setting, else `LANCE_MAX_TEMP_DIRECTORY_SIZE`, else 100 GiB. Both env vars are plain integer byte counts.
- Spill files go to the process's temporary filesystem; required spill space can exceed the right-hand side's size; exceeding the limit fails the update with a resource-exhaustion error.
- `LANCE_BYPASS_SPILLING` set: sort-merge fails instead of running unbounded in memory.
- The pool bounds DataFusion operator memory, not total RSS. `MALLOC_ARENA_MAX=4` (glibc) can lower RSS; process-wide.

### Concurrency
- "When several fragment updates run concurrently, each operation has its own external pool and spill work. Plan memory and temporary-disk capacity for their combined usage."
- Nothing on commit conflicts on this page.

---

## docs/src/guide/row_ids.md

- `_rowaddr`: physical location (fragment id + offset); changes after updates or compaction. "Do not use `_rowaddr` as a durable external key."
- `_rowid`: stable only when stable row IDs are enabled. Without them, `_rowid` "is not a persistent identifier and should not be stored for later reconciliation".
- Enable at creation only: `lance.write_dataset(table, uri, enable_stable_row_ids=True)`; check `dataset.has_stable_row_ids`.
- Limit: passing the option on later writes does not migrate an existing dataset. Migration is separate and "currently exposed in Rust as `Dataset::migrate_to_stable_row_ids`"; before it, quiesce data writers and index builds, drop secondary indexes, recreate them after.
- Read: `dataset.to_table(columns=["id", "value"], with_row_id=True)`.
- Stable ids survive `dataset.update(...)` and compaction.
- Lineage columns (need stable row ids), requested as ordinary columns: `_row_created_at_version`, `_row_last_updated_at_version`. An update keeps `_rowid` and `_row_created_at_version` and advances `_row_last_updated_at_version`; an append sets both to the append version. They are dataset versions, not timestamps.
- Secondary indexes reference physical row addresses; stable ids do not remove index maintenance. Compact first, then optimize or rebuild indexes.
- Hand-built update transactions: read rows with `with_row_id=True` and attach ids to rewritten fragments with `lance.fragment.RowIdSequence`; "Otherwise, rewritten rows can silently receive new identities." Details in `distributed_write.md#handling-stable-row-id` (not read here).
- Concurrency: only the migration quiesce rule above.

---

## docs/src/guide/migration.md

- Python API is called stable; breaking changes get deprecation warnings 1–2 months ahead. All SDKs share one release version (semver from `1.0.0`); any breaking change in Rust, Python or Java bumps the major. Majors not listed here broke only narrow or internal Rust traits.
- Package version is independent of on-disk `data_storage_version` (`2.0`, `2.1`, `2.2`, `2.3`). Upgrading the SDK does not rewrite tables.
- 15.0.0: `DirectoryNamespace` / `RestNamespace` `create_table_version`, `describe_table_version`, `batch_delete_table_versions` return `lance_namespace` models, not dicts (`response.version.version`, `model_dump()`).
- 11.0.0: `document_granularity="list_element"` needs v3 reader capability.
- 9.0.0: new FTS indexes default to format v2; code analyzer and `block_size=256` need v3; pin with `format_version` or `LANCE_FTS_FORMAT_VERSION`. Maintenance of an existing FTS index keeps its format version.
- 7.2.0: `IndexSegmentBuilder` removed; use `create_index_uncommitted`, `merge_existing_index_segments(...)`, `commit_existing_index_segments(...)`.
- 5.0.0: default data storage version changed from 2.0 to 2.1. In 2.1 `DataFile.column_indices` gives `-1` to non-leaf fields (struct and list containers); only leaves and packed structs get indices. Example `x: i32, y: [f32], z: {a: i32}`: 2.0 `[0, 1, 2, 3, 4]`, 2.1 `[0, -1, 1, -1, 2]`. Affects only users who "construct `DataFile` messages directly, for example when building operations by hand for `Dataset.commit`". Opt back with `data_storage_version="2.0"` at creation.
- 1.0.0: scalar index `SearchResult` carries a null set (`NullableRowIdSet`).
- 0.39: Python `dataset::diff_meta` removed in favor of `dataset::delta` (`DatasetDelta`: `list_transactions`, `get_inserted_rows`, `get_updated_rows`).
- No entry for 12, 13 or 14. Nothing on concurrency.

---

## docs/src/guide/tags_and_branches.md

### Tags
- `ds.tags.create("v1-prod", (None, 1))`, `ds.tags.update("v1-prod", (None, 2))`, `ds.tags.delete("v1-prod")`, `ds.tags.list()` (dict), `ds.tags.list_ordered()` (list of tuples).
- `reference` (for `create`, `update`, `checkout_version`): an int (version on the current branch), a tag name string, or a tuple `(branch_name, version)`; `(None, 2)` and `("main", 2)` both mean the default branch; `("branch", None)` is that branch's latest.
- Creating or deleting a tag does not create a dataset version.
- Tagged versions are exempt from `cleanup_old_versions()`; delete the tag first to let the version go.

### Branches
- `branch_ds = ds.create_branch("experiment")` (from the checked-out version); `ds.create_branch("new-experiment", "experiment-rc")` (from a tag).
- Write to a branch: `lance.write_dataset(data, branch_ds, mode="append")`.
- `ds.checkout_version("tag")`, `ds.checkout_version(("experiment", None))`, `ds.checkout_version(("main", None))`.
- `ds.branches.list()`, `ds.branches.delete("name")`.
- Creating or deleting a branch does not create a version. Each branch has its own linear history, so version numbers overlap across branches; use `(branch, version)` tuples.
- `"main"` is reserved; cannot be used as a name when creating, deleting or updating branches.
- A branch cannot be deleted while other branches or tags reference it.
- Branches hold references to data files; cleanup does not delete files a branch still references. Delete unused branches to free them.
- Nothing on write conflicts.

---

## docs/src/guide/data_types.md

- Lance "supports the full Apache Arrow type system". Listed: `Boolean`, `Int8..Int64`, `UInt8..UInt64`, `Float16`, `Float32`, `Float64`, `Decimal128/256`, `Date32/64`, `Time32/64`, `Timestamp`, `Duration`, `Utf8`, `LargeUtf8`, `Binary`, `LargeBinary`, `FixedSizeBinary(n)`.
- String/binary compression by field metadata: `lance-encoding:compression` = `zstd` | `lz4` | `fsst` | `none`, and `lance-encoding:compression-level`.
  - Values of 32 KiB or more are compressed one value at a time (full-zip), so a random read decodes only what it needs.
  - In stable 2.1 and 2.2 that per-value path honors only `zstd`; an `lz4` request falls back to `zstd` with a warning. 2.3 honors the request.
  - `compression-level` applies on the per-value path only with explicit `zstd`; for large text and JSON, level 1 compresses as well as the default 3 and about twice as fast.
- Vectors: `FixedSizeList` is the recommended type: `pa.field("vector", pa.list_(pa.float32(), 128))`. "Always use `FixedSizeList` (not variable-length `List`)" for storage and indexing.
- Precision: `Float32` standard; "`Float16` or `BFloat16` can reduce storage by 50% with minimal accuracy loss"; `Int8` for quantized.
- Dimensions divisible by 8 give best SIMD.
- Search: `ds.to_table(nearest={"column": "vector", "q": query_vector, "k": 10})`; with index add `"nprobes": 20`; can combine with `filter=`.
- Index above about 10,000 vectors: `ds.create_index("vector", index_type="IVF_RQ", num_partitions=256)` (recommended for most cases) or `index_type="IVF_HNSW_SQ"`.
- Variable length: `pa.list_(pa.utf8())`; `LargeList` also named.
- Struct: `pa.struct([...])`.
- Map: `pa.map_(pa.utf8(), pa.utf8())` — "Map writes require Lance file format version 2.2 or later" (`data_storage_version="2.2"`).
- Blob: new datasets use blob v2 (`lance.blob.v2`) via `from lance import blob_array, blob_field`; target file versions `>= 2.2` use Blob v2 and convert legacy blob-marked inputs. Legacy path for `<= 2.1`: `pa.field("video", pa.large_binary(), metadata={"lance-encoding:blob": "true"})`. Read whole payloads: `ds.read_blobs("video", indices=[0])` → `(row_address, payload)`; file-like lazy: `ds.take_blobs("video", indices=[0])`.
- Within the `2.x` family, append, update, merge-insert and compaction can target another exact `2.x` version without rewriting existing files; `0.1` cannot be mixed with `2.x`.
- Variant (`arrow.parquet.variant`): Rust section only; JSON columns keep their separate JSONB representation.
- Nothing on concurrency.

---

## docs/src/guide/arrays.md

- BFloat16: 16-bit, 2–3 digits of precision, range of a 32-bit float (~1e-38 to ~1e38). A plain 16-bit float has range ~5.96e-8 to 65504.
  - `from lance.arrow import bfloat16_array, BFloat16Array, PandasBFloat16Array`; `bfloat16_array([1.1, 2.1, 3.4])`; `BFloat16Array.from_numpy(np_array)` with `ml_dtypes.bfloat16`; pandas dtype string `"lance.bfloat16"`; `to_numpy` converts back.
- `lance.arrow.ImageURIArray.from_uris([...])` — URIs not strongly validated, images not read. `.read_uris()` → `lance.arrow.EncodedImageArray`.
- `EncodedImageArray.to_tensor(decoder=None)` → `lance.arrow.FixedShapeImageTensorArray`; uses Pillow if no decoder; raises if neither. `FixedShapeImageTensorArray.to_encoded()` encodes (PNG by default via Pillow). `lance.arrow.ImageArray.from_array` builds either from a `pyarrow.BinaryArray` or `pyarrow.FixedShapeTensorArray`.
- Integer tensor compression (Lance 2.3 writer, unstable): field metadata `lance-encoding:compression` = `lz4` | `zstd`; optional `lance-encoding:axis-delta` = `auto:<axis>` | `on:<axis>` | `off`. "Floating-point vectors and stable 2.1/2.2 writers keep their existing behavior."
- Nothing on concurrency.

---

## docs/src/guide/json.md

- Write: `pa.array([json.dumps(obj), ...], type=pa.json_())` as a column, then `lance.write_dataset(...)`.
- Stored as JSONB under the `lance.json` extension type; read back as Arrow's JSON type.
- Functions (in filters):
  - `json_extract(col, '$.user.name')` — JSONPath; returns JSON text, so strings keep their quotes: `filter="json_extract(data, '$.user.name') = '\"Alice\"'"`.
  - `json_get(col, key_or_index)` — returns JSONB for chaining; index is a numeric string (`'0'`).
  - `json_get_string(col, key)` — no quotes; null if conversion fails; numbers and booleans become their string form.
  - `json_get_int(col, key)` — int64; numbers truncated; parseable strings accepted; true→1, false→0; null on failure.
  - `json_get_float(col, key)` — float64; null on failure.
  - `json_get_bool(col, key)` — 0→false, non-zero→true; `"true"/"yes"`, `"false"/"no"` case-insensitive; else null.
  - `json_exists(col, '$.path')` → boolean.
  - `json_array_contains(col, '$.tags', 'python')` — matches the element's JSON text or, for a string element, its content.
  - `json_array_length(col, '$.tags')` — null if the path is missing or not an array; empty array is 0.
- Nested access: chain `json_get`, e.g. `json_get_string(json_get(json_get(data, 'user'), 'profile'), 'name') = 'Alice'`, or use `json_extract` with a path.
- Index one path:
  ```python
  from lance.indices import IndexConfig
  dataset.create_scalar_index("data", IndexConfig(index_type="json",
      parameters={"target_index_type": "btree", "path": "x"}))
  dataset.to_table(filter="json_get_int(data, 'x') = 10")
  ```
  The index matches by path literal: built with `path="x"` → query `json_get_int(data, 'x')`; built with `path="$.user.name"` → query `json_extract(data, '$.user.name')`.
- Full text over the document: `dataset.create_scalar_index("data", index_type="INVERTED", base_tokenizer="simple", lower_case=True, stem=True, remove_stop_words=True)`.
- JSON columns are not struct columns: dot notation (`meta.lang`) is for structs; JSON needs `json_get_*` / `json_extract`.
- Limits stated: "JSON functions are currently only available for filtering, not for projection in query results"; JSONPath may not support all advanced features; large documents may slow queries; flatter is faster.
- Advice: index frequently filtered paths "before creating computed columns for the same fields".
- Nothing on concurrency, on `add_columns`/`merge` with a JSON column, or on NULL JSON values.

---

## Unclear or possibly newer than 13.0.0

Checked against installed pylance 13.0.0 by signature inspection (`.venv/bin/python`, pyarrow 26.0.0):

**Not in 13.0.0 (docs are ahead):**
- `LanceFragment.update_columns` keyword arguments `strategy`, `max_hash_rows`, `max_hash_bytes`, `external_memory_pool_bytes`, `max_temp_directory_bytes`. Installed signature: `update_columns(self, data_obj, left_on='_rowid', right_on=None, schema=None, *, with_offsets=False)`. The whole of `update_join.md` (strategies, 250,000-row / 1 GiB thresholds, spill limits, env-var fallbacks) therefore describes a later release; which join 13.0.0 runs and how much memory it takes is not stated by these docs.
- `lance.CleanupPolicy`, `lance.KeepOne`, `lance.RemoveAll`, and `cleanup_old_versions(policy=..., keep_one=..., delete_concurrency=...)`. Installed signature: `cleanup_old_versions(older_than=None, retain_versions=None, *, delete_unverified=False, error_if_tagged_old_versions=True, delete_rate_limit=None, versions=None)`. The `lance.auto_cleanup.policy*` and `lance.auto_cleanup.delete_concurrency` config keys were not checked and likely go with these.

**Present in 13.0.0 (name exists; behavior not tested):**
- `LanceDataset.merge(data_obj, left_on, right_on=None, schema=None)`, `add_columns(transforms, read_columns=None, reader_schema=None, batch_size=None)`, `alter_columns`, `drop_columns`, `insert`, `update(updates, where=None, conflict_retries=10, retry_timeout=30 s, data_storage_version=None)`, `merge_insert` builder with `when_matched_update_all`, `when_not_matched_insert_all`, `when_not_matched_by_source_delete`, `data_storage_version`, `conflict_retries`, `retry_timeout`.
- `LanceOperation.Update(removed_fragment_ids, updated_fragments, new_fragments, fields_modified, fields_for_preserving_frag_bitmap, update_mode, updated_fragment_offsets)`.
- `write_dataset(..., enable_stable_row_ids=, auto_cleanup_options=, data_storage_version=)` with `"2.3"` and `"next"` accepted; `has_stable_row_ids`; `lance.fragment.RowIdSequence`.
- `optimize.compact_files(..., data_storage_version=, compaction_mode='reencode'|'try_binary_copy'|'force_binary_copy')`, `optimize.enable_auto_cleanup`, `optimize.disable_auto_cleanup`; `AutoCleanupConfig` is at `lance.dataset.AutoCleanupConfig`, not exported as `lance.AutoCleanupConfig` in 13.0.0 (the docs import `from lance import AutoCleanupConfig`).
- `tags.create/update/delete/list/list_ordered`, `create_branch`, `branches`, `checkout_version`, `explain_cleanup_old_versions`.
- `lance.blob_field`, `lance.blob_array`, `read_blobs`, `take_blobs`; `lance.batch_udf`; `lance.indices.IndexConfig`; `pa.json_()`; `lance.arrow` BFloat16 and image arrays.

**Not checked against 13.0.0 (treat as possibly newer):**
- `any_element(...)`, `array_any_match(...)` / `any_match`, and the `listelement` scalar index (the page itself calls them experimental).
- The decimal-literal comparison rule in filters.
- `skip_auto_cleanup=True` on `write_dataset` (not in the installed `write_dataset` signature printed; the page names it).
- The `asof=` fallback to an older retained version.
- The `json` scalar index type and the string-compression fallback rules (`lz4` → `zstd` on the per-value path; level 1 advice).
- `lance.compaction.data_storage_version` table-config key; mixed data-file versions inside one dataset and their feature flags.
- AxisDelta integer-tensor compression (2.3 only, unstable).

**Unclear in the docs themselves:**
- `dataset.merge`: no statement on rows whose key is absent from the new table (the practice assumes NULL), on duplicate keys, on merging a column name that already exists, or on memory use for a 3.2 M-row join. No statement on conflicts beyond the general "schema changes conflict with most concurrent writes".
- `LanceFragment.update_columns`: the page says the caller must commit, but the commit call and `LanceOperation.Update` are only in `distributed_write.md`, which was not among the pages read. Read that page before relying on the commit shape, conflict behavior or stable-row-id handling.
- `read_and_write.md` says "an update rewrites the affected columns for the affected rows" while also saying updated rows "will be removed and inserted back into the table, changing their position to the end". Which holds for `dataset.update` versus `merge_insert` versus `update_columns` is not spelled out.
- Whether `add_columns` with a reader / a `pa.json_()` field, or `alter_columns` to and from JSON, works: not stated.
- `Float16` in a `FixedSizeList` is named as a storage choice and as an `alter_columns` cast target; whether vector indexes and `nearest` accept float16 directly is not stated on these pages. Nothing is said about a 4,096-dimension limit either way.
- Stable row IDs cannot be turned on from Python for an existing dataset (Rust-only migration); whether our tables were created with them decides whether `_rowid` may be kept anywhere.
