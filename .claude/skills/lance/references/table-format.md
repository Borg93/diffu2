Covers which writes clash (the full table), what a join or update leaves on disk, versions, tags and cleanup, row ids, table config keys, blob columns, and what 13.0.0 lacks.
Read before running two writers on one table, committing a raw `LanceOperation`, cleaning up or tagging versions, saving a row id, or using a feature SKILL.md does not name.

Source: Lance docs `docs/src/format/` at 15.0.0-beta, read 2026-11-17; a tag such as (format/table/schema.md) names
the page. **[measured 13.0.0]** = seen on small throwaway tables with pylance 13.0.0. **[newer?]** = in the pages,
possibly absent from 13.0.0. A spec change lands before its implementation, so a feature on a page may be in no
library. Keep the data we keep in layouts marked stable: unstable ones get no migrations. (format/AGENTS.md)

## Which writes may run at once

- **Incompatible** = the commit fails with a non-retryable error; retrying may do something other than what was meant.
- **Retryable** = the commit fails with a retryable conflict error; the application re-reads and re-executes.
- **Rebaseable** = the commit layer rewrites the transaction against the newer version; the caller sees success.
- Rules are **directional**: stated for the operation being committed, against one another writer committed after
  this one's `read_version`.

Read each row as: "I am committing ROW; since my `read_version` someone committed COLUMN-entry".

| Committing | Incompatible (non-retryable) | Retryable | Rebaseable (auto) | Explicitly compatible |
|---|---|---|---|---|
| `Append` | `Overwrite`, `Restore`, `UpdateMemWalState` | none | none | everything else, **including `Append`** ("multiple writers can append without worry about conflicts") |
| `Delete` | `Overwrite`, `Restore`, `UpdateMemWalState` | `Merge`, `Rewrite`, `DataReplacement` (each only if overlapping fragments) | `Delete`, `Update` (deletion masks merged; **retryable if both touched the same rows**) | |
| `Overwrite` | none ("we do not care what has happened since the read version") | `UpdateConfig` (only if same config key), `Overwrite` (always), `UpdateMemWalState` (always) | | `Append` and the rest |
| `CreateIndex` | `Overwrite`, `Restore`, `UpdateMemWalState` | `Rewrite` (only if overlapping fragments, no stable row ids, no fragment reuse index); `DataReplacement` (only if the replaced column is being indexed and `data_change` is not false) | `CreateIndex` (only if both modify the same singleton index, e.g. fragment reuse index, mem wal index) | `Append`, `Update`, `Delete`, `CreateIndex` |
| `Rewrite` | `Overwrite`, `Restore` | **`Merge` (always)**; `DataReplacement`, `Delete`, `Update` (each only if overlapping fragments); `Rewrite` (overlapping fragments, or both carry a fragment reuse index); `CreateIndex` (overlapping fragments and no fragment reuse index or stable row ids) | `CreateIndex` writing the fragment reuse index while the `Rewrite` carries one | `Append` (not listed) |
| `Merge` | `Overwrite`, `Restore`, `UpdateMemWalState`, **`Project`** | **`Update`, `Append`, `Delete`, `Merge`, `Rewrite`, `DataReplacement` — all "always"** | none | `CreateIndex`, `UpdateConfig`, `ReserveFragments` (not listed) |
| `Project` | `Overwrite`, `Restore`, `UpdateMemWalState` | `Project` (always), `Merge` (always) | | the rest |
| `Restore` | `UpdateMemWalState`; `UpdateConfig` (only if it updates schema or field metadata) | | | everything else ("trumps any other operation") |
| `ReserveFragments` | `Overwrite`, `Restore` | | | the rest |
| `Clone` | cannot conflict (only valid as first operation; fails if a dataset exists) | | | |
| `Update` | `Overwrite`, `Restore` | `Rewrite` (only if overlapping fragments), `DataReplacement` (only if overlapping fragments), **`Merge` (always)** | `Delete`, `Update` | `Append` (not listed) |
| `UpdateConfig` | "conflict" (class not stated): `Overwrite`; `Restore` (only if the `UpdateConfig` updates schema or field metadata); `UpdateConfig` (only if both modify the same config) | | | the rest |
| `DataReplacement` | `Overwrite`, `Restore`, `UpdateMemWalState`; `Delete` (only if it removes a target fragment outright); `Update` (only if it removes a target fragment outright); `Project` (only if it drops a field being replaced). The last three become **retryable** when `data_change` is false | `DataReplacement` (same field and overlapping fragments); `CreateIndex` (replaced field being indexed and `data_change` not false); `Rewrite` (overlapping fragments); `Update` (rewrites rows out of a target fragment, or rewrites a replaced field in place); `Merge` (always) | | `Delete`/`Update` that only add a deletion vector to a target fragment; `Append` |
| `DataOverlay` | `Overwrite`, `Restore`, `UpdateMemWalState` | `Rewrite` (overlapping fragments); `Merge` (always); a row-moving `Update` (any update that is not `REWRITE_COLUMNS`) touching an overlaid fragment | | `DataOverlay` (any fields), `Append`, `Delete`, `REWRITE_COLUMNS` update, `DataReplacement` |
| `UpdateBases` | only `UpdateBases` with the same base id, name or path | | | everything else |
| `CompositeOperation` | vs any **named** operation: fails closed (rejected as a conflict, not compared). vs another `CompositeOperation`: computed from the actions | | | |

(format/table/transaction.md)

Reading the table:

- Python call to operation: `merge` and `add_columns` are `Merge`; `optimize.compact_files()` is `Rewrite`;
  `update()` is `Update`; dropping a column is `Project`.
- `Append` after a `Merge` succeeds (measured), yet the `Merge` fails if it commits second. A failed `Merge` throws
  its computed column data away; the files stay for cleanup.
- `Delete` after `Merge` is "only if overlapping fragments", but a `Merge` names every fragment: in practice always.
- Two `Update`s rewriting the same field of the same fragment (`REWRITE_COLUMNS`): unspecified. The page says only
  that `Update`/`Update` rebases by merging deletion masks. Give column updates of one table a single writer.
- `UpdateMemWalState` is absent from the `Update` and `Rewrite` lists while present in `Append`, `Delete`,
  `CreateIndex`, `Merge`, `Project`; the page does not say whether that is intended.

Two handles opened at the same version on a 10-row, 2-fragment local table; one commits, then the other
**[measured 13.0.0]**:

| Second writer (stale handle) | After a concurrent | Result |
|---|---|---|
| `LanceDataset.commit(Append, read_version=old)` | `Append` | ok |
| same | `add_columns` (`Merge`) | ok; appended rows read `NULL` in the new column |
| same | `update()` | ok |
| same | `optimize.compact_files()` (`Rewrite`) | ok |
| `add_columns(...)` | `Append` | `CommitConflictError`: "Retryable commit conflict ... This Merge transaction was preempted by concurrent transaction Append" |
| `merge(...)` | `Append` | same error text, raised as `OSError` |
| `add_columns(...)` | `update()` | retryable conflict (preempted by `Update`) |
| `add_columns(...)` | `add_columns` | retryable conflict (preempted by `Merge`) |
| `add_columns(...)` | compaction | retryable conflict (preempted by `Rewrite`) |
| `optimize.compact_files()` | `Append` | ok |
| `optimize.compact_files()` | `add_columns` | retryable conflict (preempted by `Merge`); rewritten fragments left for GC |
| `optimize.compact_files()` | `update()` on the same fragment | retryable conflict (preempted by `Update`) |
| `update(...)` (high-level) | `Append` | ok |
| `update(...)` (high-level) | `add_columns` | **ok** — the high-level `update()` re-executes instead of surfacing the retryable conflict the page describes |
| `update(...)` / `delete(...)` (high-level) | compaction of the same fragments | **ok**, same reason |

- The high-level `update` and `delete` retry inside the call (measured), so a "retryable" outcome of the page does
  not always surface as an error. `merge_insert` is listed with them but was not measured.
- Not measured: a raw `LanceDataset.commit(LanceOperation.Update(..., fields_modified=[...]))` against a concurrent
  `Merge`, `Rewrite` or `Update`. By the page it is retryable against `Merge` (always) and against `Rewrite` on
  overlapping fragments, and the caller of a raw commit redoes the work itself.

Commit mechanics (format/table/transaction.md):

- One writer wins each version number; the loser checks the transactions committed since its `read_version`,
  rebases if it can, and retries: `LanceDataset.commit(..., max_retries=20)`.
- Every attempt writes `_transactions/{read_version}-{uuid}.txn`; a failed attempt leaves that and its data files
  unreferenced until cleanup.
- Checked before every commit and retry: a **required field** (it and all its ancestors non-nullable) has data in
  every fragment the transaction adds or whose data files it changes; every data file stores at least one field;
  fragment ids are unique.

## What each operation does

`lance.LanceOperation` in 13.0.0 exposes `Append`, `Clone`, `CreateIndex`, `DataOverlay`, `DataReplacement`, `Delete`,
`Merge`, `Overwrite`, `Project`, `Restore`, `Rewrite`, `Update`, `UpdateConfig`. Absent: `ReserveFragments`,
`UpdateMemWalState`, `UpdateBases`, `CompositeOperation`. (format/table/transaction.md)

- `CreateIndex`: adds, replaces or removes indices. Uncovered fragments are unindexed. A second index of the same
  name replaces the first.
- `Rewrite`: compaction, defragmentation, re-ordering; values unchanged, **row addresses change**. Its new fragment
  ids are reserved first with `ReserveFragments`.
- `Merge`: adds columns and changes the schema. "The set of fragments provided in the operation will be the final set
  of fragments in the resulting dataset." The page calls it "overly general" and prefers `Rewrite`,
  `DataReplacement` or `Append`.
- `Project`: drops columns; metadata only, the data stays in the files.
- `Clone`: shallow (metadata only, through `base_paths`) or deep; only as a dataset's first operation.
- `Update`: changes values, adds and removes no rows.
  `Update(removed_fragment_ids, updated_fragments, new_fragments, fields_modified, fields_for_preserving_frag_bitmap, update_mode, updated_fragment_offsets)`.
  - `REWRITE_ROWS` deletes the rows and writes them into **new fragments** (few rows, or most columns, change). Row
    addresses change, and `_rowid` too unless the table was created with stable row ids.
  - `REWRITE_COLUMNS` rewrites the columns inside their fragments by tombstoning the old column (most rows, few
    columns, change).
- `DataReplacement`: replaces the data files behind some fields of existing fragments; rows stay put. One
  `DataReplacementGroup` = one target fragment + one new data file, applied to the fragment as it stands at commit:
  a data file with the same field ids, order and file version has its path, size and base replaced; if no data file
  holds any of the new file's fields (a column added as all nulls) the new file is appended; if the fragment's files
  hold all of them, those are tombstoned and the new file appended (a file left with no live field is removed;
  legacy V1 files: exact match only); anything else is rejected.
  - `data_change` (absent = true) **[newer?]**: whether 13.0.0 has the flag is unknown; `DataReplacementGroup`
    exists. True withdraws index coverage of the target fragments on the replaced fields and, with stable row ids,
    sets last-updated-at for **every row** of the targets. False asserts identical values and does neither.

## What a join does on disk

- A fragment holds several data files, each a subset of columns named by **field ids** (`DataFile.fields`,
  `DataFile.column_indices`). Adding or backfilling a column attaches a data file to each fragment; nothing is
  rewritten. (format/table/index.md)
- A field with no data file in a fragment reads all `NULL` there. So the joined column is nullable, and a later
  in-place update covers the fragments appended since the join as well.
- A column rewrite sets the old column's field id to `-2` (**tombstone**) in its data file and appends a new data
  file. The old bytes stay until compaction, so repeated updates grow the table and the data files per fragment.
- Changing a field's data withdraws index coverage of that field for the touched fragments (those rows are scanned).
- A `REWRITE_COLUMNS` update is positional against a fragment. Plan it against the version you commit against: a
  plan made before a committed compaction names wrong fragment ids (a concurrent one is a retryable conflict).
- `fields_modified` takes field ids. Look the id up by name in the current schema at each use: ids are sparse (an id
  is not the column's position), and the id of a dropped field may be reused. (format/table/schema.md)
- Field ids by change: add = new id; rename, reorder, metadata or nullability change = same id; **cast = new field,
  new id**; drop = field leaves the schema, data stays in the files.
- At creation ids run depth-first from 0, or come from Arrow field metadata `lance:field_id` (unique, `< 2^30`). A
  schema copied from another table carries that table's ids: strip the metadata unless that is intended.
- **Unenforced primary key**: field metadata `lance-schema:unenforced-primary-key` = `true`/`1`/`yes`, optional
  `lance-schema:unenforced-primary-key:position` (1-based). Lance checks no uniqueness; merge-insert can enforce it.
  The field and its ancestors are non-nullable, a leaf, outside any list or map. **Fixed once set.**
- Deletions: at most one deletion file a fragment a version; `physical_rows` counts deleted rows. Materializing
  deletions (compaction) changes row addresses and needs an index remap or rebuild.
- Keep the fragment count low (large appends, periodic compaction): the manifest lists every fragment and is
  rewritten whole on every commit, so commit and open cost grow with the count. Fragment ids are never reused, not
  after deletion or restore. Writers compete for the next version even when changing different fragments.
  (format/table/fragment_metadata.md)

## Versions, tags and cleanup

- Every commit is a new manifest and version number; old versions stay readable until cleaned up.
- Explicit cleanup in 13.0.0:
  `cleanup_old_versions(older_than, retain_versions, *, delete_unverified, error_if_tagged_old_versions=True, delete_rate_limit, versions)`
  and `explain_cleanup_old_versions`. Automatic cleanup is off unless `lance.auto_cleanup.interval` is non-zero.
- The format pages leave unspecified what cleanup removes and how tags block it (they point to
  `guide/read_and_write.md#automatic-cleanup`, not read for this file). Stated: files reachable from a retained
  manifest stay; protected are "the latest version, retained history, tags, and branch references under the table's
  cleanup policy". (format/table/blob.md)
- A **tag** is `_refs/tags/{name}.json` at the dataset root: it names (branch, version), creates no version, and
  protects that version from cleanup. Names: alphanumerics, `.`, `-`, `_`; no `/`; no leading or trailing `.`; no
  `..`; not ending `.lock`. (format/table/branch_tag.md)
- A **branch** is a shallow clone under `tree/{branch_name}/` with its own version numbers. **No branch merge
  exists**: use a branch for a diverging copy only. Field ids are unique per branch, so diverged branches can assign
  the same id. Names: segments of alphanumerics, `.`, `-`, `_`; no leading or trailing `/`, `//`, `..`, `\`; not
  ending `.lock`; not `main`; case-sensitive. 13.0.0 has `tags`, `branches`, `create_branch`, `shallow_clone`,
  `deep_clone`.
- Copying the whole dataset root gives a working table (all paths inside are relative). A **shallow clone** points
  at the source's files through `base_id`: keep the source (tag it) while the clone needs it; copying the clone's
  root alone leaves the source's data behind. (format/table/layout.md)
- `_versions/latest_version_hint.json` is optional and safe to delete; whether 13.0.0 writes it is unknown.
- **Feature flags**: a library that meets an unknown bit in `reader_feature_flags` or `writer_feature_flags` returns
  "unsupported", so a table touched by a newer library using a new feature can become unreadable or unwritable for
  13.0.0. Bits 0 to 12 are listed on the page; 13 `FLAG_INDEPENDENT_COVERING_FIELDS`, 14 `FLAG_MANAGED_BLOBS` (sticky),
  15 `FLAG_NON_REUSABLE_FIELD_IDS`, 16 `FLAG_ZONE_WRITE_SEEDS`, 17 `FLAG_OBJECT_METADATA`,
  18 `FLAG_ROW_ADDR_INDICES`, 19 `FLAG_INLINE_DELETIONS`. Bits 13 and 15 to 19 are likely after 13.0.0; the page
  declares 16 and 17 reserved and unsupported. (format/table/versioning.md)

## Row ids

(format/table/row_id_lineage.md)

- **Row address** `_rowaddr` = `(fragment_id << 32) | local_row_offset`. It changes on compaction and on row-moving
  updates. Indices refer to rows by address.
- **Row id** `_rowid`: with stable row ids off (the default) it equals the row address, "not stable and should not
  be used as a persistent identifier". Join on `image`; a saved `_rowid` or `_rowaddr` is valid only on a table
  created with stable row ids.
- Stable row ids: a unique `u64` from the manifest counter `next_row_id`, constant for the logical row's life. On a
  commit conflict the writer reassigns, so appended rows' ids are final only at commit.
- Enable at creation: `write_dataset(..., enable_stable_row_ids=True)`. "An ordinary write with
  `enable_stable_row_ids = true` does not migrate an existing dataset." Migration is Rust
  `Dataset::migrate_to_stable_row_ids` (drop every secondary index first, quiesce writers, one merge commit, no
  retry) **[newer?]**: no such method seen in 13.0.0, which has `has_stable_row_ids`.
- With stable row ids (or a fragment reuse index) compaction stops clashing with `CreateIndex`.
- Only with stable row ids, from creation: `_row_created_at_version`, `_row_last_updated_at_version`, giving "which
  rows changed since version N" by filter:
  - inserted: `_row_created_at_version > begin AND _row_created_at_version <= end`
  - updated: `_row_created_at_version <= begin AND _row_last_updated_at_version > begin AND _row_last_updated_at_version <= end`
- Those two names and `_rowid` are reserved column names.
- Spilling row id sequences over ~200 KB to hidden data-file columns needs `lance.row_lineage.spill` and bit 11
  **[newer?]**.

## Table config keys

`Manifest.config` is a string map stored with every version; `lance.` is reserved for Lance and unrecognized keys
are ignored. Write it with `update_config()` (an `UpdateConfig` transaction). It is distinct from schema metadata
(attached to the scan output schema) and table metadata (arbitrary). 13.0.0 has `config`, `update_config`,
`delete_config_keys`, `metadata`, `update_metadata`, `schema_metadata`, `update_schema_metadata`,
`update_field_metadata`. (format/table/table_config.md)

An unrecognized `lance.compaction.*` key is ignored with a warning, so a typo or a too-new key fails silently. Keys
marked * may be newer than 13.0.0.

- `lance.auto_cleanup.`: `interval` (commits between runs; unset or `0` = off), `older_than` (duration string:
  `3600s`, `2h`), `retain_versions` (kept regardless of age), `referenced_branch` (also remove old versions reachable
  only through a branch or tag reference; default `false`), `delete_rate_limit` (deletes a second).
- `lance.compaction.`: `target_rows_per_fragment` (default 1,048,576; smaller fragments are candidates),
  `max_rows_per_group` (1024), `max_bytes_per_file`, `materialize_deletions` (`true`),
  `materialize_deletions_threshold` (`0.1`; `0` = all; above `1.0` = never), `defer_index_remap` (`true`: the
  fragment reuse index is updated instead of remapping indices), `batch_size`, `io_buffer_size`.
- `lance.compaction.` *: `index_remap_mode` (`direct` default, `merged`), `compaction_mode` (`re-encode`,
  `try-binary-copy`, `force-binary-copy`; overrides deprecated `enable_binary_copy`/`enable_binary_copy_force`),
  `binary_copy_read_batch_bytes` (16 MiB), `max_source_fragments`, `max_source_rows`, `max_source_bytes` (caps per
  run, default unlimited; bytes exclude blob v2 payloads), `max_overlays_per_fragment` (`10`, or `none`).

## Blob columns

(format/table/blob.md)

- A pointer table (a string path column) needs no blob column.
- Blob v2 needs **file format 2.2 or later**; scans read a small descriptor a row, payloads on demand.
- Released kinds: `0 Inline`, `1 Packed` (shared sidecar), `2 Dedicated` (own sidecar,
  `data/{data_file_stem}/{encoded_blob_id}.blob`), `3 External`. External objects are not owned by Lance: never
  deleted by it, mutation goes undetected (no ETag or hash), and a missing object fails only at read.
- `write_dataset` 13.0.0: `external_blob_mode: 'reference' | 'ingest'` (reference = External, no bytes copied;
  ingest = bytes copied into Lance storage), `allow_external_blob_outside_bases`, `blob_pack_file_size_threshold`,
  `data_storage_version` up to `'2.3'`.
- Reusing an existing object (compaction, reference appends, shallow clones) needs the source snapshot protected,
  e.g. by a tag, until the destination commit completes; an open handle pins nothing.
- **Managed blobs** (kinds `4 Managed`, `5 ManagedWithBase`, `_blobs/{uuid}.blob`) **[newer?]**: schema metadata
  `lance.blob.managed = "true"` sets bit 14 **permanently** (removing the opt-in, dropping the blob fields or a
  restore do not clear it). Before enabling, upgrade or stop **all** maintenance
  clients: an older client's cleanup can delete sidecars still referenced. Whether 13.0.0 writes them is unknown.

## Unusable or absent in 13.0.0

- **Data overlay files** (cell-level new values, flag bit 6): "currently experimental and not yet supported in any
  library"; release builds reject the flag unless opted in. 13.0.0 still exposes `LanceOperation.DataOverlay`,
  `DataOverlayFile`, `DataOverlayGroup`: leave them unused on tables we keep. (format/table/data_overlay_file.md)
- **`CompositeOperation`** ("Transaction V2", actions such as `AssertCellsUnchanged`) **[newer?]**: absent from
  13.0.0; "Readers before v14.0.0 fail on a `CompositeOperation`" when decoding the transaction (scanning still
  works), which aborts the older writer's commit.
- **MemWAL** (`_mem_wal/`, `UpdateMemWalState`; experimental streaming-write layer): not usable from 13.0.0 as far
  as the pages and the `LanceOperation` list show. Batch appends from one or two processes use plain `Append`.
  (format/table/mem_wal.md)
- **Fragment tree** (`FLAG_FRAGMENT_TREE`, `_bt/`) **[newer?]**: "Support for creating or reading tree tables is not
  yet available in released Lance versions."
- **Clustering definitions** (`Manifest.clustering`, `max_clustering_id`) and the **fragment metadata map**
  (`DataFragment.metadata`) **[newer?]**: need bit 17, "Reserved, unsupported until writers implement the
  maintenance rules". (format/table/clustering.md)
- **Non-reusable field ids** (`max_allocated_field_id` + bit 15) **[newer?]**: reached only by an explicit one-way
  migration commit; under it ids are never reused and an overwrite gives every field a new id.
- **Inline deletion vectors** (`DeletionFile.inline`, bit 19) **[newer?]**.
