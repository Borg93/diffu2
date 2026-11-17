# Lance: write, evolve and type a table

Covers: filtering, appending, adding and updating columns, deleting, compaction, cleanup, tags, row ids, data file versions, column types and JSON, as the Lance guide pages state them.
Read when: you are about to write one of these calls and SKILL.md does not spell it out.

Marks: **[absent]** = not in pylance 13.0.0 (the docs are ahead); **[unchecked]** = not checked against 13.0.0, treat
as possibly newer. The last section lists the names confirmed in 13.0.0.

## Filter rows (guide/read_and_write.md)

- Operators: `>`, `>=`, `<`, `<=`, `=`, `AND`, `OR`, `NOT`, `IS [NOT] NULL`, `IS [NOT] TRUE`, `IS [NOT] FALSE`,
  `IN`, `LIKE`, `NOT LIKE`, `regexp_match(column, pattern)`, `CAST`. `IN` lists take brackets: `label IN [10, 20]`.
- Nested access by subscript: struct by field name (`note['email']`), list by index. Backtick names holding special
  characters or SQL keywords, each path segment on its own. Limit: field names containing `.` are unsupported.
- Typed literals: `date '2021-01-01'`, `timestamp '2021-01-01 00:00:00'` (`timestamp(0|3|6|9)` = s/ms/us/ns,
  default 6), `decimal(8,3) '1.000'` (the decimal comparison rule is [unchecked]).
- `CAST` names: `float`→Float32, `double`→Float64, `int`→Int32, `bigint`→Int64, `string`→Utf8, `binary`→Binary,
  `boolean`→Boolean, `date`→Date32.
- `any_element(list_col, 'pred')` / `array_any_match(col, d -> ...)` on `list<struct>` and the `listelement` scalar
  index that answers them: experimental, syntax may change, [unchecked].
- An in-place method (`delete`, `update`, ...) advances its own handle. Reopen or check out to latest every other
  open handle: it keeps reading its pinned snapshot.

## Append rows (guide/read_and_write.md)

- `lance.write_dataset(data, uri)` takes a `pyarrow.Table`, `pandas.DataFrame`, `pyarrow.dataset.Dataset` or
  `Iterator[pyarrow.RecordBatch]`; an iterator needs `schema=`.
- `ds.insert(new_table)` or `lance.write_dataset(new_table, ds, mode="append")` leave existing fragments unrewritten.

## Add a column (guide/data_evolution.md)

Run alone: beside an append, one of the two fails, which one depends on order.

- Pre-computed values: `dataset.merge(new_data, "image")`, `new_data` holding the join column and the new columns.
  Installed: `merge(data_obj, left_on, right_on=None, schema=None)`. The page shows only a table holding every key.
  Unstated: what rows with an absent key get (the practice assumes NULL), duplicate keys, a column name that
  already exists, memory for a 3.2 M-row join.
- Empty column: `dataset.add_columns(pa.field("embedding", pa.list_(pa.float32(), 128)))` or a `pa.schema([...])`;
  metadata-only. File format `<= 2.1` cannot add a sub-column under an existing struct; `2.2` can, also inside lists.
- Filled by SQL: `dataset.add_columns({"hash": "sha256(name)", "status": "'active'"})`; a literal sets all rows.
- Filled by a UDF: decorate a function `RecordBatch -> RecordBatch | DataFrame` (called once a batch) with
  `@lance.batch_udf(checkpoint_file="x.sqlite")` and pass it to `add_columns`. A failed run restarts from the
  checkpoint. Pitfall: the file "can get quite large, since it needs to store unsaved results for up to an entire
  data file".
- Installed: `add_columns(transforms, read_columns=None, reader_schema=None, batch_size=None)`. Unstated: whether
  `add_columns` with a reader or a `pa.json_()` field works.

## Update values

- By SQL: `dataset.update({"age": "age + 2"}, where="name = 'Blob'")`. Values are SQL expressions, so a string
  literal needs inner quotes (`{"name": "'Bob'"}`); no `where` = all rows. Installed: `update(updates, where=None,
  conflict_retries=10, retry_timeout=30 s, data_storage_version=None)`. For a set of individual rows use one
  `merge_insert`: an `update` per row in a Python loop "works, but is inefficient". (guide/read_and_write.md)
- By key: `dataset.merge_insert("image").when_matched_update_all().execute(new_table)`; add
  `.when_not_matched_insert_all()` for an upsert; `.when_not_matched_insert_all()
  .when_not_matched_by_source_delete("age >= 40")` replaces a filtered region. The builder also takes
  `conflict_retries`, `retry_timeout`, `data_storage_version`. It hash-joins, so key on a unique column and expect
  relative order to change. (guide/read_and_write.md)
- A held column in place: `LanceFragment.update_columns`. Installed: `update_columns(self, data_obj,
  left_on='_rowid', right_on=None, schema=None, *, with_offsets=False)`. It returns new fragment metadata and the
  ids of the modified fields; the caller commits them with `LanceOperation.Update(removed_fragment_ids,
  updated_fragments, new_fragments, fields_modified, fields_for_preserving_frag_bitmap, update_mode,
  updated_fragment_offsets)`. Read `distributed_write.md#update-columns` before relying on the commit shape, conflict
  behaviour or stable-row-id handling: it was not among the pages read. (guide/update_join.md)
  - Matched rows take the supplied values, unmatched rows keep theirs; an empty right-hand side is a no-op.
  - The right-hand side holds the join column and at least one *existing* column: this call adds no new column.
  - Join-key types match on both sides. `right_on` is removed from the payload; when `left_on` differs, a
    replacement column named `left_on` updates the left key. `_rowid` and `_rowaddr` are never replacement columns.
  - Concurrent fragment updates each take their own memory pool and spill space: plan for the sum.
- [absent] `update_columns(strategy="auto"|"hash"|"sort_merge", max_hash_rows=, max_hash_bytes=,
  external_memory_pool_bytes=, max_temp_directory_bytes=)`; which join 13.0.0 runs and its memory use are unstated.
  Later release: `auto` hash-joins up to 250,000 rows and 1 GiB estimated allocation (give both thresholds or
  neither), else sort-merges; `hash` is in memory, unbounded; `sort_merge` spills within the pool
  (`LANCE_MEM_POOL_SIZE`, else 256 MiB) and the temp limit (`LANCE_MAX_TEMP_DIRECTORY_SIZE`, else 100 GiB;
  exceeding it fails the update; spill can exceed the right-hand side's size); `LANCE_BYPASS_SPILLING` makes it
  fail rather than run unbounded. The pool bounds operator memory, not RSS (`MALLOC_ARENA_MAX=4` can lower RSS).
- Unclear in the docs: read_and_write.md says both "an update rewrites the affected columns for the affected rows"
  and that updated rows are removed and re-inserted at the end; which holds for `update`, `merge_insert` and
  `update_columns` is not spelled out.

## Delete rows; drop, rename or cast a column

- `dataset.delete("name = 'Bob'")` marks rows in a deletion file; indices stay valid. (guide/read_and_write.md)
- `dataset.drop_columns(["name"])` is metadata-only: the data stays on disk until `optimize.compact_files(...)` then
  `cleanup_old_versions()`, after which it may be unrecoverable. Tag before the drop and delay cleanup. Nested
  removal (`people.item.city`) needs format `2.2`. (guide/data_evolution.md)
- Rename: `dataset.alter_columns({"path": "id", "name": "new_id"})`; nested path `"meta.id"`.
- Cast: `dataset.alter_columns({"path": "embedding", "data_type": pa.list_(pa.float16(), 128)})` rewrites that
  column only. Pitfall: "If the column has an index, the index will be dropped if the column type is changed."
  Unstated: casts to and from JSON.

## Compact (guide/read_and_write.md)

- `dataset.optimize.compact_files(target_rows_per_fragment=1024 * 1024)` skips fragments above the count and merges
  the rest in fragment-id order, so row order holds; deleted rows are removed. Also takes `data_storage_version=`,
  `compaction_mode='reencode'|'try_binary_copy'|'force_binary_copy'`.
- It writes a new version and leaves old versions and files in place.
- Compact first, then `optimize_indices()` or rebuild: compaction changes `_rowaddr` of the rows it moves.

## Clean up old versions (guide/read_and_write.md)

- `dataset.cleanup_old_versions()` removes versions older than two weeks; `older_than=timedelta(days=1)` overrides.
  Installed: `cleanup_old_versions(older_than=None, retain_versions=None, *, delete_unverified=False,
  error_if_tagged_old_versions=True, delete_rate_limit=None, versions=None)`. `explain_cleanup_old_versions`
  previews.
- Kept always: the current version (`older_than=timedelta(0)` is safe), tagged versions, branch dependencies.
- A removed version is gone: `version=N` fails; `lance.dataset(uri, asof=t)` returns the latest *retained* version
  before `t`, `ValueError` only when none predates it ([unchecked]).
- Files left by aborted or failed commits go once 7 days old. Guardrail: pass `delete_unverified=True` only when
  "no other process is operating on the dataset at all"; a concurrent write may lose its files, "leaving the
  dataset corrupted".
- Auto cleanup runs in the commit path: `write_dataset(..., auto_cleanup_options=AutoCleanupConfig(interval=20,
  older_than_seconds=3600))`, `ds.optimize.enable_auto_cleanup(...)` / `disable_auto_cleanup()`, or
  `ds.update_config({"lance.auto_cleanup.interval": "20", "lance.auto_cleanup.older_than": "3600s"})`. Import
  `AutoCleanupConfig` from `lance.dataset`: 13.0.0 does not export it from `lance`. `skip_auto_cleanup=True` on
  `write_dataset` is [unchecked] (not in the installed signature printed).
- [absent] `lance.CleanupPolicy`, `KeepOne`, `RemoveAll`, `cleanup_old_versions(policy=, keep_one=,
  delete_concurrency=)`; the `lance.auto_cleanup.policy*` and `.delete_concurrency` config keys likely go with them
  ([unchecked]).

## Tag or branch (guide/tags_and_branches.md)

- `ds.tags.create("v1-prod", (None, 1))`, `.update`, `.delete`, `.list()` (dict), `.list_ordered()` (tuples). Delete
  the tag to let cleanup take its version.
- A reference (`create`, `update`, `checkout_version`) is an int (version on the current branch), a tag name, or
  `(branch, version)`; `(None, 2)` and `("main", 2)` mean the default branch; `("branch", None)` is its latest.
- `ds.create_branch("experiment")` branches from the checked-out version, `ds.create_branch("new", "tag")` from a
  tag; write with `lance.write_dataset(data, branch_ds, mode="append")`; `ds.branches.list()` / `.delete(name)`.
- Tags and branches create no dataset version. Version numbers overlap across branches: use the tuple form.
- `"main"` is reserved. A branch stays undeletable while a tag or branch references it, and cleanup keeps every
  file a branch references: delete unused branches to free them.

## Keep row identity (guide/row_ids.md)

- Store `image` as the durable key. `_rowaddr` (fragment id + offset) changes after updates and compaction;
  `_rowid` persists only with stable row IDs.
- Stable row IDs are set at creation only: `lance.write_dataset(table, uri, enable_stable_row_ids=True)`; check
  `dataset.has_stable_row_ids`. The option on a later write migrates nothing, and the migration is Rust-only
  (`Dataset::migrate_to_stable_row_ids`: quiesce writers and index builds, drop secondary indexes, recreate after).
  Whether our tables were created with them decides whether `_rowid` may be kept anywhere.
- With them: `to_table(columns=[...], with_row_id=True)`; ids survive `update` and compaction; the lineage columns
  `_row_created_at_version` and `_row_last_updated_at_version`, requested as ordinary columns, hold dataset
  versions (an update advances only the second; an append sets both).
- Indexes reference physical addresses either way: compact first, then optimize or rebuild.
- In a hand-built update transaction, read with `with_row_id=True` and attach the ids to rewritten fragments with
  `lance.fragment.RowIdSequence`: "Otherwise, rewritten rows can silently receive new identities."
  (`distributed_write.md#handling-stable-row-id`, not read.)

## Choose a data file version

- `data_storage_version` is set by create and overwrite and is the default for later writes; append, update,
  merge-insert and compaction leave it unchanged. Upgrading pylance rewrites nothing. (guide/read_and_write.md)
- One operation may target another exact `2.x` version without rewriting existing files:
  `ds.update({...}, data_storage_version="2.2")`, `merge_insert(...)....data_storage_version("2.2")`,
  `optimize.compact_files(data_storage_version="2.2")`. `0.1` never mixes with `2.x`. First upgrade every reader and
  writer: the commit sets feature flags that stay set, and older clients reject such snapshots. Mixed versions,
  their flags and the `lance.compaction.data_storage_version` config key are [unchecked].
- Name exact versions: `"stable"` and `"next"` resolve per release. `"2.3"` is unstable (a later revision may fail
  to read its files): experiments only.
- Since 5.0.0 the default is 2.1, where `DataFile.column_indices` gives `-1` to struct and list containers
  (`x: i32, y: [f32], z: {a: i32}`: 2.0 `[0, 1, 2, 3, 4]`, 2.1 `[0, -1, 1, -1, 2]`); it matters when building
  `DataFile` messages by hand for `Dataset.commit`. (guide/migration.md)
- Since 9.0.0 new FTS indexes default to format v2; the code analyzer and `block_size=256` need v3; pin with
  `format_version` or `LANCE_FTS_FORMAT_VERSION`; maintenance keeps an index's format. The page has no entry for
  12, 13 or 14. (guide/migration.md)

## Choose column types (guide/data_types.md, guide/arrays.md)

- Vectors: `FixedSizeList`, never variable-length `List`. "`Float16` or `BFloat16` can reduce storage by 50% with
  minimal accuracy loss"; dimensions divisible by 8 give the best SIMD. The page indexes above about 10,000 vectors
  with `IVF_RQ` (`num_partitions=256`, its pick for most cases) or `IVF_HNSW_SQ`. Unstated: whether vector indexes
  and `nearest` accept float16 directly, and any 4,096-dimension limit.
- BFloat16 (`lance.arrow.bfloat16_array`, `BFloat16Array.from_numpy` with `ml_dtypes.bfloat16`): 2-3 digits of
  precision over the range of a 32-bit float; a plain float16 spans about 5.96e-8 to 65504.
- String and binary compression by field metadata: `lance-encoding:compression` = `zstd`|`lz4`|`fsst`|`none`,
  `lance-encoding:compression-level`. Values of 32 KiB or more are compressed one at a time, so a random read
  decodes only its own. On that path 2.1 and 2.2 honour only `zstd` (`lz4` falls back with a warning) and the level
  applies only with explicit `zstd`; for large text and JSON, level 1 compresses as well as the default 3 about
  twice as fast. [unchecked]
- `pa.map_(...)` needs `data_storage_version="2.2"`.
- Blobs: file versions `>= 2.2` use blob v2 (`from lance import blob_array, blob_field`) and convert legacy inputs;
  `<= 2.1` uses `pa.large_binary()` with metadata `{"lance-encoding:blob": "true"}`. `ds.read_blobs(col,
  indices=[0])` returns `(row_address, payload)`; `ds.take_blobs` is lazy and file-like.

## Store and filter JSON (guide/json.md)

- Write `pa.array([json.dumps(obj), ...], type=pa.json_())`; stored as JSONB, read back as Arrow JSON. Struct dot
  notation (`meta.lang`) does not reach into JSON: use the functions below.
- `json_extract(col, '$.user.name')` returns JSON text, so strings keep their quotes:
  `filter="json_extract(data, '$.user.name') = '\"Alice\"'"`.
- `json_get(col, key_or_index)` returns JSONB for chaining; an index is a numeric string (`'0'`):
  `json_get_string(json_get(json_get(data, 'user'), 'profile'), 'name') = 'Alice'`.
- `json_get_string`: no quotes; numbers and booleans become their string form; null on failure.
- `json_get_int`: int64; numbers truncated, parseable strings accepted, true→1, false→0; null on failure.
- `json_get_float`: float64; null on failure.
- `json_get_bool`: 0→false, non-zero→true; `"true"/"yes"`, `"false"/"no"` case-insensitive; else null.
- `json_exists(col, '$.path')`; `json_array_contains(col, '$.tags', 'python')` matches an element's JSON text or a
  string element's content; `json_array_length(col, '$.tags')` is null for a missing path or a non-array, 0 for an
  empty array.
- Index one path ([unchecked]): `dataset.create_scalar_index("data", lance.indices.IndexConfig(index_type="json",
  parameters={"target_index_type": "btree", "path": "x"}))`. Query with the path literal the index was built with:
  `path="x"` → `json_get_int(data, 'x')`; `path="$.user.name"` → `json_extract(data, '$.user.name')`. The page
  advises indexing a filtered path before making a computed column for it.
- Full text over the document: `dataset.create_scalar_index("data", index_type="INVERTED",
  base_tokenizer="simple", lower_case=True, stem=True, remove_stop_words=True)`.
- Limits: JSONPath may lack advanced features; large documents slow queries, flatter is faster. Unstated: NULL
  JSON values, and `merge` with a JSON column.

## Names confirmed in 13.0.0

By signature inspection only, nothing run against data: `merge`, `add_columns`, `alter_columns`, `drop_columns`,
`insert`, `update`, the `merge_insert` builder, `LanceOperation.Update`, `write_dataset(enable_stable_row_ids=,
auto_cleanup_options=, data_storage_version=)` with `"2.3"` and `"next"` accepted, `has_stable_row_ids`,
`lance.fragment.RowIdSequence`, `optimize.compact_files`, `enable_auto_cleanup`, `disable_auto_cleanup`, `tags.*`,
`create_branch`, `branches`, `checkout_version`, `explain_cleanup_old_versions`, `blob_field`, `blob_array`,
`read_blobs`, `take_blobs`, `lance.batch_udf`, `lance.indices.IndexConfig`, `pa.json_()`, `lance.arrow` BFloat16.
