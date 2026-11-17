# Lance table format: what a Python user must know

Source: Lance docs at 15.0.0-beta, `docs/src/format/` (read in full 2026-11-17). Installed library: pylance 13.0.0.
Rules below are from the pages unless marked **[measured 13.0.0]** (small throwaway tables, see the last-but-one section)
or **[newer?]** (see "Unclear or possibly newer than 13.0.0").

Vocabulary used throughout:

- **Incompatible** = commit fails, non-retryable error; retrying may do something different from what was meant.
- **Retryable** = commit fails with a retryable conflict error; the application re-reads and re-executes.
- **Rebaseable** = the commit layer rewrites the transaction against the newer version and commits; the caller sees success.
- Conflict rules are **directional**: they are stated for the operation being committed, against an operation that
  another writer committed after this one's `read_version`.

---

## docs/src/format/index.md

- Lance is a stack of specs: file format, table format, index formats, catalog specs, namespace client spec.
- File format: column data in large pages, no row groups; statistics and search structures live in indices, not files.
- Table format is two-dimensional: rows are grouped into **fragments**; each fragment holds **several data files**,
  each supplying a subset of columns. Adding or backfilling a column is mainly a metadata operation, not a rewrite.
- Indices are table objects, versioned and committed transactionally with the table.
- Catalogs: Directory Catalog (no infrastructure) and REST Catalog. Not needed for a plain local/object-store path.

## docs/src/format/AGENTS.md (and the repo-root AGENTS.md it points to)

- Contributor rules, not user rules. What matters to a reader of the spec:
  - Format changes need a PMC vote; **a spec change lands in its own PR and the implementation follows later**. So a
    feature being in these pages does not mean any library implements it.
  - A format marked **stable** is a compatibility contract (backward and forward). A format marked **unstable** is
    disposable: no migrations, no compatibility with earlier unstable revisions. Do not store data we keep in an
    unstable/experimental layout.
  - Python and Java bindings are thin wrappers over the Rust core; parameter names match across bindings.

## docs/src/format/table/index.md

How the pieces fit:

- **Manifest** = one immutable description of one version: full schema, list of fragments, monotonically increasing
  version number, optional index section, `config` map (`lance.`-prefixed keys reserved), plus table metadata and
  schema metadata maps. Every commit writes a new manifest; nothing is edited in place.
- **Fragment** = horizontal slice of rows. Unique `uint32` id, assigned incrementally from the dataset's maximum
  fragment id. Holds one or more data files plus at most one deletion file. `physical_rows` counts deleted rows too.
- **Data file** = Lance file holding a subset of the fragment's columns, identified by **field ids**
  (`DataFile.fields`, `DataFile.column_indices`). Rules:
  - Each data file of a fragment should hold a distinct set of field ids.
  - A schema field with no data file in a fragment reads as entirely `NULL` in that fragment.
  - Field id `-2` in a data file is a **tombstone**: that column of that file is ignored. A column rewrite tombstones
    the old column and appends a new data file to the fragment. The old bytes stay on disk until the file is rewritten.
  - All negative field ids are reserved: `-1` unassigned (in memory only), `-2` tombstone, `-3` `_rowid`,
    `-4` `_row_created_at_version`, `-5` `_row_last_updated_at_version`.
  - File format 2.0: every field, including struct/list containers, has a column index. 2.1+: non-leaf fields have
    `-1` in `column_indices`.
- **Deletion file** (deletion vector) = the 0-based row offsets deleted in one fragment. At most one per fragment per
  version. Two formats: Arrow IPC `.arrow` (flat Int32Array, sparse deletions) and Roaring bitmap `.bin` (dense).
  Readers must filter those offsets. Materializing deletions (rewriting data files without the rows) changes row
  addresses and requires index remap/rebuild.
- **Data evolution** = add column / update column / drop column with backfill by attaching data files to existing
  fragments; no full-table rewrite.
- **Unenforced primary key**: field metadata `lance-schema:unenforced-primary-key` = `true`/`1`/`yes`, optional
  `lance-schema:unenforced-primary-key:position` (1-based). Lance does not validate uniqueness; merge-insert can
  enforce it. Field and all ancestors must be non-nullable, a leaf, not inside a list or map. **Fixed once set; cannot
  be updated or removed.**
- `max_allocated_field_id` (optional manifest field): when set, field ids are never reused. See schema.md. **[newer?]**
- `DataFragment.metadata` (`map<string,string>`): needs `FLAG_OBJECT_METADATA` (`1 << 17`), reserved/unsupported. **[newer?]**
- File metadata size hints (`DataFile.file_metadata_size_bytes`): advisory read optimization only.
- Data overlay files: "currently experimental and not yet supported in any library" (feature flag 64).
- External manifest store: only needed when the object store lacks atomic put/rename.

## docs/src/format/table/schema.md

- Logical type strings map to Arrow: `string`/`large_string`, `binary`/`large_binary`, `halffloat`/`float`/`double`,
  `decimal:128:P:S`, `date32:day`, `timestamp:<unit>:<tz>` (`-` = no timezone), `struct`, `list`, `list.struct`,
  `large_list`, `fixed_size_list:<type>:<size>`, `fixed_size_binary:<size>`, `dict:<value>:<key>:<ordered>`, `map`
  (unordered keys only), extension `blob`, `json`, `fixed_size_list:lance.bfloat16:<n>`.
  `Utf8View`/`BinaryView` map to `string`/`binary`.
- **Field ids**: every field, nested included, has a unique integer id. At creation: depth-first from 0, or taken
  from Arrow field metadata `lance:field_id` if supplied (must be unique, `< 2^30`; **a schema copied from another
  table carries that table's ids**, so strip the metadata unless intended). Nested fields carry `parent_id`
  (`-1` = top level).
- Properties: preserved for the life of the field; unique within a version; **sparse** (not consecutive, so a field
  id is not the column's position).
- New fields on an existing table:
  - Legacy (no `max_allocated_field_id`): writer picks the next id from fields the current version still references;
    **the id of a dropped field may be reused**.
  - Non-reusable mode (`max_allocated_field_id` + `FLAG_NON_REUSABLE_FIELD_IDS`, both or neither): ids always above
    the high-water mark; caller cannot choose ids; only reachable by an explicit migration commit; one-way; restore to
    a pre-activation version must fail. **[newer?]**
- Field ids are unique per dataset **and per branch**; two diverged branches can assign the same id. "Lance does not
  yet merge branches."
- Schema evolution in terms of ids:
  - Add column: new id. Drop column: field removed from schema (data stays in files). Rename, reorder, metadata or
    nullability change: id unchanged.
  - **Type replacement (cast)**: a new field with a new id; the old identity is retired.
  - Overwrite: with `max_allocated_field_id` set, every field gets a new id even if names and types are unchanged.
- Field metadata: `lance-encoding:` prefix = encoding/compression; `ARROW:extension:` = extension types.
  Clustering hint `unenforced_clustering_key_position` (1-based; 0 = not a clustering key; may be nullable).
- Blob v2 field: one physical column per blob field, one field id; the descriptor's children are not dataset fields
  and do not move `max_allocated_field_id`.

## docs/src/format/table/transaction.md

### Commit protocol

- MVCC, optimistic concurrency. Each commit atomically writes the next manifest using **rename-if-not-exists** or
  **put-if-not-exists**; exactly one writer wins a given version number.
- Manifest naming: V1 `{version}.manifest`; V2 `{u64::MAX - version:020}.manifest` (latest sorts first).
  pylance: `enable_v2_manifest_paths` (default `True` in `write_dataset` 13.0.0).
- Each attempt writes a transaction file `_transactions/{read_version}-{uuid}.txn` (or inlines the transaction in the
  manifest). A transaction has `read_version`, `uuid`, `operation`.
- Loser of the race: loads the transactions committed since its `read_version`, checks each for conflict, rebases if
  possible, retries. pylance `LanceDataset.commit(..., max_retries=20)`.
- Validation before every commit (and again on every retry):
  - A **required field** (it and all ancestors non-nullable) must have data in every fragment the transaction adds or
    whose data files it changes; making a field required checks every fragment.
  - Every data file stores at least one field. Fragment ids unique. Field-id allocation rules hold.
- External manifest store: 4 steps (stage `{version}.manifest-{uuid}`, reserve in the KV store, copy to the final
  path = commit point, update pointer). Only for stores without atomic primitives; a reader unaware of the store may
  be one commit behind.

### Operations (exact names)

| Operation | What it does |
|---|---|
| `Append` | Adds new fragments; existing data untouched. Fragment ids assigned at manifest construction, not at transaction creation. |
| `Delete` | Marks rows deleted via deletion vectors; may update fragments or remove whole fragments. Carries `predicate`. |
| `Overwrite` | Creates or completely replaces the table: data, schema, config. |
| `CreateIndex` | Adds, replaces or removes indices. Uncovered fragments are simply unindexed. Same name twice: the second replaces the first. |
| `Rewrite` | Reorganizes without changing values: compaction, defragmentation, re-ordering. **Changes row addresses.** New fragment ids must be reserved first with `ReserveFragments`. |
| `Merge` | Adds new columns and changes the schema. **"The set of fragments provided in the operation will be the final set of fragments in the resulting dataset."** The page calls it "overly general" and says to prefer `Rewrite`, `DataReplacement` or `Append`. |
| `Project` | Drops columns; metadata only, data files untouched. |
| `Restore` | Reverts the table to a previous version (as a new version). |
| `ReserveFragments` | Pre-allocates fragment ids (advances the max fragment id). |
| `Clone` | Shallow (metadata only, via `base_paths`) or deep copy. Only as the first operation of a dataset. |
| `Update` | Changes row values, no rows added or removed. Two modes: **`REWRITE_ROWS`** deletes the rows in their fragments and writes them into **new fragments** (best when few rows or most columns change); **`REWRITE_COLUMNS`** rewrites the affected columns inside the fragments by tombstoning the old column (best when most rows but few columns change). |
| `UpdateConfig` | Changes table config, table metadata, schema metadata or field metadata. |
| `DataReplacement` | Replaces the data files backing some fields of existing fragments, rows not moved. One `DataReplacementGroup` = one target fragment + one new data file. |
| `DataOverlay` | Attaches overlay files (cell-level new values). Experimental. |
| `UpdateMemWalState` | Updates MemWAL index state. No compatibility section of its own on the page. |
| `UpdateBases` | Adds base paths. |
| `CompositeOperation` | "Transaction V2": ordered list of `UserStep`s of granular `Action`s applied atomically. **[newer?]** |

pylance 13.0.0 `lance.LanceOperation` exposes: `Append`, `Clone`, `CreateIndex`, `DataOverlay`, `DataReplacement`,
`Delete`, `Merge`, `Overwrite`, `Project`, `Restore`, `Rewrite`, `Update`, `UpdateConfig`. Not exposed:
`ReserveFragments`, `UpdateMemWalState`, `UpdateBases`, `CompositeOperation`.
`Update(removed_fragment_ids, updated_fragments, new_fragments, fields_modified, fields_for_preserving_frag_bitmap, update_mode, updated_fragment_offsets)`.

`DataReplacement` detail (how a group applies, in order, to the fragment as it stands at commit):

- A data file of the fragment with the same field ids in the same order and file version: its path/size/base are replaced.
- No data file of the fragment holds any of the new file's fields (column added as all nulls): the new file is appended.
- The fragment's files hold all of the new file's fields: those fields are tombstoned (`-2`) and the new file appended;
  a file left with no live field is removed from the fragment. (Legacy V1 files: exact match only.)
- Anything else is rejected.
- `data_change` (absent = true). True: withdraws index coverage of the target fragments on the replaced fields,
  tombstones older overlays on those fields, sets last-updated-at version of every row in the targets (stable row ids).
  False: writer asserts identical values, none of that happens. **[newer?]** (the `data_change` flag)

### Conflict rules, per operation being committed

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

Pair summary for our workload (both directions):

| Pair | A committing after B | B committing after A |
|---|---|---|
| `Append` / `Append` | ok | ok |
| `Append` / `Merge` | `Append` after `Merge`: **ok** (the appended fragments read `NULL` for the new column) | `Merge` after `Append`: **retryable** |
| `Append` / `Update` | ok | ok |
| `Append` / `Rewrite` | ok | ok |
| `Merge` / `Merge` | retryable | retryable |
| `Merge` / `Update` | `Merge` after `Update`: retryable | `Update` after `Merge`: retryable |
| `Merge` / `Delete` | `Merge` after `Delete`: retryable (always) | `Delete` after `Merge`: retryable (overlapping fragments) |
| `Merge` / `Rewrite` | retryable | retryable |
| `Merge` / `Project` | `Merge` after `Project`: **incompatible** | `Project` after `Merge`: retryable |
| `Update` / `Update` | rebaseable; same rows modified by both: retryable | same |
| `Update` / `Delete` | rebaseable; same rows: retryable | same |
| `Update` / `Rewrite` | `Update` after `Rewrite`: retryable if overlapping fragments | `Rewrite` after `Update`: retryable if overlapping fragments |
| `Delete` / `Rewrite` | retryable if overlapping fragments | retryable if overlapping fragments |
| `Rewrite` / `Rewrite` | retryable if overlapping fragments (or both carry a fragment reuse index) | same |
| `CreateIndex` / `Rewrite` | retryable if overlapping fragments and neither stable row ids nor a fragment reuse index | same |
| anything / `Overwrite` or `Restore` committed first | incompatible (except `Overwrite` itself, `Restore`, `UpdateBases`, `Clone`) | |

### Rebase mechanism (what "rebaseable" means in practice)

- `TransactionRebase` keeps the fragments as of the read version, the set of modified/deleted fragment ids, and for
  `Delete`/`Update` the `affected_rows`.
- Two deletes/updates on the same fragment with **non-overlapping `affected_rows`**: deletion vectors are merged and
  rewritten, commit proceeds.
- `Update` against a concurrent compaction of its fragment: retryable, not rebaseable (the fragment no longer exists);
  the application re-executes against the new version.
- `Delete` against a concurrent `Restore`: incompatible.

### Action-based transactions (`CompositeOperation`) **[newer?]**

- "Readers before v14.0.0 fail on a `CompositeOperation`" when they decode the transaction (history, conflict check);
  scanning still works. For conflict checking the failure aborts the older writer's commit ("the safe outcome").
- Writers commit one only when the calling code opts in per commit.
- Actions are deltas with local placeholders for new ids; `data_unchanged` per action (default false).
- Without assertions this is **snapshot isolation**; a writer whose values depend on what it read must declare
  `AssertCellsUnchanged` (`UPDATE t SET a = b` and `UPDATE t SET b = a` otherwise both commit).
- Fragment reuse history changes only through `AppendFragReuseTransition` and `TrimFragReuse`.
- An unknown operation in reserved field numbers 100-199 must be treated as conflicting with everything.

## docs/src/format/table/table_config.md

- `Manifest.config`: `map<string,string>` stored with every version. `lance.` prefix reserved for Lance; other
  libraries use their own prefix (e.g. `lancedb.`). Unrecognized keys are ignored.
- Read/modify through `Dataset.update_config()` (a normal transaction, `UpdateConfig`). Distinct from **schema
  metadata** (attached to scan output schema) and **table metadata** (arbitrary).
  pylance 13.0.0 has `config`, `update_config`, `delete_config_keys`, `metadata`, `update_metadata`,
  `schema_metadata`, `update_schema_metadata`, `update_field_metadata`.

| Key | Type | Meaning |
|---|---|---|
| `lance.auto_cleanup.interval` | non-negative int | Commits between automatic cleanup runs; runs on the commit whose version is a multiple of it. Unset or `0` = disabled. |
| `lance.auto_cleanup.older_than` | duration string (`3600s`, `2h`) | Only versions older than this are removed. |
| `lance.auto_cleanup.retain_versions` | non-negative int | Minimum number of most-recent versions kept regardless of age. |
| `lance.auto_cleanup.referenced_branch` | bool | Whether auto cleanup also removes old versions only reachable through a branch or tag reference. Default `false`. |
| `lance.auto_cleanup.delete_rate_limit` | non-negative int | Max delete operations per second. |
| `lance.compaction.target_rows_per_fragment` | non-negative int | Fragments with fewer rows are compaction candidates. Default 1,048,576. |
| `lance.compaction.max_rows_per_group` | non-negative int | Default 1024. |
| `lance.compaction.max_bytes_per_file` | non-negative int | Max bytes per rewritten data file. |
| `lance.compaction.materialize_deletions` | bool | Rewrite fragments with deletions to drop the rows. Default `true`. |
| `lance.compaction.materialize_deletions_threshold` | float 0.0–1.0 | Deleted fraction that triggers it. Default `0.1`; `0` = all; above `1.0` = never. |
| `lance.compaction.defer_index_remap` | bool | `true`: indices not remapped during compaction; the fragment reuse index is updated instead. |
| `lance.compaction.index_remap_mode` | `direct` or `merged` | Default `direct`. |
| `lance.compaction.batch_size` | non-negative int | Scan batch size during compaction. |
| `lance.compaction.io_buffer_size` | non-negative int | Bytes queued in the I/O buffer. |
| `lance.compaction.compaction_mode` | `re-encode`, `try-binary-copy`, `force-binary-copy` | Overrides deprecated `enable_binary_copy`/`enable_binary_copy_force`. |
| `lance.compaction.binary_copy_read_batch_bytes` | non-negative int | Default 16 MiB. |
| `lance.compaction.max_source_fragments` | non-negative int | Cap per run (incremental compaction). Default unlimited. |
| `lance.compaction.max_source_rows` | non-negative int | Cap on live source rows per run. Default unlimited. |
| `lance.compaction.max_source_bytes` | non-negative int | Cap on source bytes per run (excludes blob v2 payloads). Default unlimited. |
| `lance.compaction.max_overlays_per_fragment` | non-negative int or `none` | Overlays a fragment may carry before full compaction. Default `10`. |

- Unrecognized `lance.compaction.*` keys are ignored with a warning.
- Mentioned on other pages, not in this table: `lance.row_lineage.spill` (opt in to spilled row lineage) and schema
  metadata `lance.blob.managed = "true"` (schema metadata, not table config).

## docs/src/format/table/clustering.md

- `Manifest.clustering`: repeated immutable definitions (`id`, `columns` as field ids, `provider`, `version`);
  `Manifest.max_clustering_id` is a never-lowered high-water mark. Fragments and index segments may carry
  `clustering_id`.
- Requires `FLAG_OBJECT_METADATA` (`1 << 17`) in `writer_feature_flags`, which versioning.md lists as
  **"Reserved, unsupported until writers implement the maintenance rules"**. Not usable today. **[newer?]**
- Readers may ignore clustering metadata.
- `Field.unenforced_clustering_key_position` / `Field.unenforced_clustering_key` are legacy column-level hints, kept.

## docs/src/format/table/versioning.md

- Two flag words in the manifest: `reader_feature_flags`, `writer_feature_flags`. A reader/writer that sees a flag it
  does not know must return an "unsupported" error for the read or write. This is how an old library is kept from
  damaging a table written by a newer one, so **a table touched by a newer library with a new feature can become
  unreadable or unwritable for 13.0.0**.

| Bit | Flag | Reader | Writer | Meaning |
|---|---|---|---|---|
| `2^0` | `FLAG_DELETION_FILES` | yes | yes | Fragments may have deletion files. |
| `2^1` | `FLAG_STABLE_ROW_IDS` | yes | yes | Row ids stable across moves and updates. |
| `2^2` | `FLAG_USE_V2_FORMAT_DEPRECATED` | no | no | Deprecated. |
| `2^3` | `FLAG_TABLE_CONFIG` | no | yes | Table config present. |
| `2^4` | `FLAG_BASE_PATHS` | yes | yes | Multiple base paths (shallow clones, multi-base). |
| `2^5` | `FLAG_DISABLE_TRANSACTION_FILE` | no | yes | Transactions recorded in the manifest, no separate file. |
| `2^6` | `FLAG_UNSTABLE_DATA_OVERLAY_FILES` | yes | yes | Data overlay files. "Unstable: release builds reject it unless explicitly opted in." |
| `2^7` | `FLAG_COVERED_INDEX_METADATA` | yes | yes | Some index declares covering columns. |
| `2^8` | `FLAG_MIXED_DATA_FILE_VERSIONS` | yes | yes | V2 data files with different exact versions in one snapshot. |
| `2^9` | `FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS` | yes | yes | Stable row ids plus a fragment reuse index. |
| `2^10` | `FLAG_FRAGMENT_REUSE_INDEX` | yes | yes | Fragment reuse index with tagged transitions. |
| `2^11` | `FLAG_SPILLED_ROW_LINEAGE` | yes | yes | Row id / version sequences stored as hidden data-file columns. Opt-in via table config. |
| `2^12` | `FLAG_FRAGMENT_TREE` | yes | yes | Fragment records live in a fragment tree; `Manifest.fragments` empty. |
| `2^13` | `FLAG_INDEPENDENT_COVERING_FIELDS` | yes | yes | Requires bit 7. |
| `2^14` | `FLAG_MANAGED_BLOBS` | yes | yes | Managed blob objects; sticky once set. |
| `2^15` | `FLAG_NON_REUSABLE_FIELD_IDS` | no | yes | `max_allocated_field_id` set. |
| `2^16` | `FLAG_ZONE_WRITE_SEEDS` | yes | yes | "Reserved and unsupported until implemented." |
| `2^17` | `FLAG_OBJECT_METADATA` | no | yes | Clustering / fragment metadata / index segment metadata. "Reserved, unsupported". |
| `2^18` | `FLAG_ROW_ADDR_INDICES` | no | yes | Stable-row-id table with an index that stores physical row addresses. |
| `2^19` | `FLAG_INLINE_DELETIONS` | yes | yes | A deletion vector stored in the manifest (`DeletionFile.inline`), no file under `_deletions/`. |

- Bits 16 and 17 reserved but unsupported; bits 20+ unknown.
- Committing schema metadata `lance.blob.managed = "true"` activates Managed Blobs permanently (not cleared by
  removing the opt-in, dropping blob fields, or restore). All maintenance clients must support it first.
- **This page does not describe version numbers, time travel, or cleanup of old versions** beyond flags. What the
  format pages say about versions and cleanup, collected:
  - Every commit = a new manifest = a new version number; old versions stay readable until cleaned up.
  - Cleanup is governed by the `lance.auto_cleanup.*` keys (off unless `interval` is set and non-zero) or run
    explicitly (pylance 13.0.0: `cleanup_old_versions(older_than, retain_versions, *, delete_unverified, error_if_tagged_old_versions=True, delete_rate_limit, versions)`, `explain_cleanup_old_versions`).
  - Protected snapshots (blob.md wording): "the latest version, retained history, tags, and branch references under
    the table's cleanup policy". Files reachable from a retained manifest must not be removed.
  - A failed commit attempt leaves unreferenced files (data files, `_transactions/*.txn`) that only cleanup removes.

## docs/src/format/table/branch_tag.md

Tags:

- A tag is a JSON file `_refs/tags/{name}.json` at the dataset root, always at root level whatever branch it names.
- Fields: `branch` (optional; null/absent = default branch), `version`, `createdAt`, `updatedAt` (RFC 3339, optional),
  `manifestSize`, `metadata` (optional string map).
- Name rules: non-empty; only alphanumerics, `.`, `-`, `_`; cannot start or end with `.`; cannot end with `.lock`;
  no `..`. **No `/` in tag names.**
- A tag is not a commit: it creates no table version. It names (branch, version) and protects that version from
  cleanup (see `lance.auto_cleanup.referenced_branch`, `error_if_tagged_old_versions`).

Branches:

- Branch metadata `_refs/branches/{name}.json` (`/` in the name encoded `%2F`). Fields: `parentBranch` (null = default
  branch), `parentVersion`, `createAt` (unix seconds), `manifestSize`, `metadata`.
- A branch dataset is a **shallow clone** of the source, stored under `tree/{branch_name}/` with its own `_versions/`,
  `_transactions/`, `_deletions/`, `_indices/`. Its version numbers are its own history.
- Name rules: non-empty; no leading/trailing `/`; no `//`; no `..` or `\`; segments only alphanumerics, `.`, `-`, `_`;
  cannot end with `.lock`; cannot be `main` (virtual name of the default branch; no `main.json` exists).
  Case-sensitive.
- **No branch merge exists** (schema.md: "Lance does not yet merge branches"). A branch is for a diverging copy, not
  for work to be folded back.
- pylance 13.0.0 exposes `tags`, `branches`, `create_branch`, `shallow_clone`, `deep_clone`.

## docs/src/format/table/data_overlay_file.md

- Status on the page: **"currently experimental and not yet supported in any library"**; feature flag 64
  (`FLAG_UNSTABLE_DATA_OVERLAY_FILES`), reader and writer; release builds reject it unless explicitly opted in.
  **Do not use** for tables we keep, even though pylance 13.0.0 exposes `LanceOperation.DataOverlay`. **[newer?]**
- What it is: a third way to change data in place, next to deletion files (remove rows) and data evolution (add or
  rewrite whole columns). An overlay supplies new values for a subset of `(row offset, field)` **cells** of a fragment
  without rewriting base data files.
- Resolution: walk the fragment's overlays newest to oldest; the first that covers the cell wins; else the base value
  (or `NULL`). Order by `committed_version` (the version the commit produced, re-stamped on retry), then by position
  in `DataFragment.overlays`. A covered cell holding `NULL` overrides to `NULL`. Deletions win over overlays.
- Coverage bitmaps index **physical row offsets** (deleted rows included). Dense = one `shared_offset_bitmap` for all
  fields; sparse = `FieldCoverage` with one bitmap per field. Value position = rank of the offset in the bitmap.
- Indices: fragment stays covered; rows covered by an overlay newer than `index.dataset_version` are excluded from
  index results and re-evaluated on the flat path (field-aware, includes `covering_fields`).
- Compaction: overlay→overlay (contiguous in `committed_version`, result takes the max) and overlay→base (a column
  rewrite; must rebuild the index or drop the fragment from its coverage in the same commit). A row-rewriting
  compaction folds overlays into the new base. Trigger: `lance.compaction.max_overlays_per_fragment`.
- Does not need stable row ids. Updates `last_updated_at_version` of covered rows when lineage is on.
- Writer support section is a stub: sparse overlays as one file are not writable by the current v2 writer.

## docs/src/format/table/blob.md

- Blob v2 = a column of nullable large binary values (images, audio, video) stored as a small **descriptor** per row;
  scans do not fetch payloads; values or byte ranges are read on demand. Requires **file format 2.2 or later**.
- Logical (manifest) schema: a struct with Arrow field metadata `ARROW:extension:name = "lance.blob.v2"` and children
  exactly `data: large_binary`, `uri: string` (minimal) or those plus `position: u64`, `size: u64` (complete).
  Per non-null value exactly one of `data` and `uri` is non-null; `position`/`size` both or neither, only with `uri`.
- Data file schema: packed struct `kind: u8`, `position: u64`, `size: u64`, `blob_id: u32`, `blob_uri: string`.
- Descriptor kinds: `0 Inline` (in the data file), `1 Packed` (shared sidecar of the data file), `2 Dedicated` (own
  sidecar), `3 External` (not owned by Lance; never deleted by Lance; no ETag/hash, so mutation is undetected; a
  missing object fails only at read), `4 Managed`, `5 ManagedWithBase` (Lance-owned objects under
  `_blobs/{uuid}.blob`, independent of data files).
- Sidecars: `data/{data_file_stem}/{encoded_blob_id}.blob`.
- Writer modes for URI input: **reference mode** (becomes External, no bytes copied; URIs outside registered bases
  need explicit permission) and **ingest mode** (bytes copied into Lance-owned storage).
  pylance 13.0.0 `write_dataset`: `external_blob_mode: 'reference' | 'ingest'`, `allow_external_blob_outside_bases`,
  `blob_pack_file_size_threshold`, `data_storage_version` up to `'2.3'`.
- Managed storage: opt in with schema metadata `lance.blob.managed = "true"`; sets `FLAG_MANAGED_BLOBS` (`1 << 14`)
  in both flag words **permanently**. Under it, compaction turns Packed/Dedicated into Managed descriptors pointing at
  the original sidecar without reading the bytes. "Before enabling Managed storage, upgrade or stop **all**
  maintenance clients": an older client's cleanup can delete adopted sidecars still referenced. **[newer?]**
- Lifetime: payloads must be complete before the manifest commit. Reusing an existing object (compaction, reference
  appends, shallow clones) requires the source snapshot to stay protected, e.g. by a tag, until the destination commit
  completes; an open handle does not pin it. Any live range retains the whole object.
- Usable today: Blob v2 with Inline/Packed/Dedicated/External is the released behaviour (the page says Packed and
  Dedicated "remain the default out-of-line storage kinds"). For pointer tables (a string path column), a blob column
  is not needed.

## docs/src/format/table/fragment_metadata.md

- **Not usable.** "Experimental ... This layout is unstable." "Support for creating or reading tree tables is not yet
  available in released Lance versions." Flag 4096 `FLAG_FRAGMENT_TREE`. Flat manifests remain the default.
  **[newer?]**
- What it would change: today every manifest carries the complete fragment list, so **commit and open cost grow with
  the number of fragments**, even when a change touches few. The tree keeps fragment records in immutable Lance leaves
  keyed by fragment id under `_bt/` (`root/{uuid}.root`, `node/{uuid}.node`, `leaf/{uuid}.lance`), with pending
  mutations (`upsert_fragment`, `remove_fragment`, `add_data_file`, `remove_data_file`, `add_deletion_file`,
  `clear_deletion_file`, `replace_data_file`) above the leaves.
- Rules that also hold for flat tables: `Manifest.max_fragment_id` is the allocator; it never decreases; **fragment
  ids are never reused**, not after deletion and not after restore; first id is 0. Writers still "compete for the next
  dataset version, even when changing different fragments".
- `DeletionFile.inline` (deletion vector stored in the manifest) needs `FLAG_INLINE_DELETIONS` (`2^19`). **[newer?]**
- Consequence for us now: keep the fragment count low (large appends, periodic compaction), because the flat manifest
  is rewritten whole on every commit.

## docs/src/format/table/layout.md

```
{dataset_root}/
    data/*.lance                     data files
    data/{data_file_stem}/*.blob     blob sidecars
    _blobs/{UUID}.blob               managed blob payloads
    _versions/*.manifest             one per version
    _versions/latest_version_hint.json   optional {"version": N}; safe to delete
    _transactions/*.txn              {read_version}-{uuid}.txn
    _deletions/*.arrow | *.bin       {fragment_id}-{read_version}-{id}.{ext}
    _indices/{UUID}/...
    _refs/tags/*.json
    _refs/branches/*.json
    _bt/...                          fragment tree objects (experimental)
    tree/{branch_name}/...           branch datasets
```

- One dataset root per dataset. All paths inside are relative, so **copying the whole root directory gives a working
  table** with no manifest change.
- `base_paths` in the manifest (`BasePath { id, name, is_dataset_root, path }`): data files, deletion files and index
  metadata may carry `base_id` to live elsewhere (hot/cold tiering, multi-region, shallow clones). Absent `base_id` =
  dataset root.
- **Shallow clone** = new manifest in a new location whose inherited fragments point at the source through `base_id`;
  new writes go to the clone. The source must be kept (a tag) for as long as the clone needs it; copying only the
  clone's root does not copy the source's data.
- Data file names: 50 characters from a UUID (24 binary digits + 26 hex), not meaningful.

## docs/src/format/table/mem_wal.md

- Title says "(Experimental)". An LSM layer over a base table for high-rate streaming writes: writers append to
  **shards**; each shard has an in-memory MemTable, a WAL (`_mem_wal/{shard_id}/wal/{bit_reversed_position}.arrow`,
  Arrow IPC), flushed **SSTables** (each a small one-fragment Lance dataset at
  `_mem_wal/{shard_id}/{random8}_gen_{i}/`), and a shard manifest (`.../manifest/{bit_reversed_version}.binpb`).
  Background compaction merge-inserts SSTables into the base table in ascending generation order.
- Base table carries one system index `__lance_mem_wal` (`MemWalIndexDetails`: `sharding_specs`,
  `maintained_indexes`, `maintain_all_indexes`, `writer_config_defaults`, `compacted_sstables`, `index_catchup`).
- Needs an unenforced primary key for lookups and last-write-wins upserts; append-only tables may omit it. All rows of
  one primary key must map to one shard. Sharding transforms: `unsharded`, `bucket` (`num_buckets` in `[1, 1024]`),
  `identity`.
- One active writer per shard, fenced by `writer_epoch`. Deletes are `_tombstone` rows. Readers must merge base table
  + SSTables (+ MemTables for strong consistency); a plain reader of the base table does not see uncompacted data.
- Interaction with ordinary transactions: `UpdateMemWalState` is **incompatible** with `Append`, `Delete`,
  `CreateIndex`, `Merge`, `Project`, `DataReplacement`, `DataOverlay`, `Restore`, and retryable against `Overwrite`.
- Usable today from pylance 13.0.0: **no** as far as these pages and the Python `LanceOperation` list show
  (`UpdateMemWalState` is not exposed). Not needed for batch appends from one or two processes; plain `Append` is
  conflict-free.

## docs/src/format/table/row_id_lineage.md

- **Row address** = `(fragment_id << 32) | local_row_offset`, exposed as `_rowaddr`. Physical. **Changes on compaction
  and on row-moving updates.** Indices reference rows by row address ("Work to support stable row IDs in indices is in
  progress").
- **Row id** (`_rowid`):
  - Stable row ids **disabled (default)**: `_rowid` equals the row address; "not stable and should not be used as a
    persistent identifier".
  - Stable row ids **enabled**: unique auto-incrementing `u64` from the manifest counter `next_row_id`, constant for
    the life of the logical row. An update writes a new physical row with the same `_rowid` and tombstones the old
    one. On a commit conflict the writer re-reads `next_row_id` and reassigns (so appended rows' ids are only final at
    commit).
- Enabling: at creation (`enable_stable_row_ids=True` in `write_dataset`), or by a migration. **"An ordinary write
  with `enable_stable_row_ids = true` does not migrate an existing dataset."** Migration is
  `Dataset::migrate_to_stable_row_ids` in Rust: requires dropping every secondary index first, quiescing writers, is a
  single merge commit that does not retry. **[newer?]** (no migration method seen in pylance 13.0.0; it has
  `has_stable_row_ids`).
- Row version tracking, **only with stable row ids**: `_row_created_at_version`, `_row_last_updated_at_version`.
  Change data feed by plain filters:
  - inserted: `_row_created_at_version > begin AND _row_created_at_version <= end`
  - updated: `_row_created_at_version <= begin AND _row_last_updated_at_version > begin AND _row_last_updated_at_version <= end`
- A `DataReplacement` with `data_change` true sets last-updated-at of **every row in the target fragments**, not only
  rows whose value changed.
- Row id sequences live in the fragment metadata (`RowIdSequence`: Range, RangeWithHoles, RangeWithBitmap, sorted
  array, unsorted array); sequences over ~200 KB may be spilled to hidden data-file columns (field ids `-3`, `-4`,
  `-5`) only if the table sets `lance.row_lineage.spill`; needs `FLAG_SPILLED_ROW_LINEAGE` (bit 11). **[newer?]**
- Reserved column names a writer must reject: `_rowid`, `_row_created_at_version`, `_row_last_updated_at_version`.
- When they matter for us:
  - Our key is the string `image`, so joins and merges should use `image`, never a saved `_rowid`/`_rowaddr`, unless
    the table was created with stable row ids.
  - With stable row ids (or a fragment reuse index), `Rewrite` (compaction) no longer conflicts with `CreateIndex`.
  - "Which rows changed since version N" needs stable row ids from creation.

---

## Checked on pylance 13.0.0 [measured 13.0.0]

Two handles opened at the same version on a 10-row, 2-fragment local table; one commits, then the other.

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

Not measured: a raw `LanceDataset.commit(LanceOperation.Update(..., fields_modified=[...]))` against a concurrent
`Merge`/`Rewrite`/`Update`. By the page it is retryable against `Merge` (always) and against `Rewrite` on overlapping
fragments, and the caller of a raw commit must redo the work itself.

## Rules for our workload (derived from the above)

- **Two processes may append at once.** `Append` conflicts only with `Overwrite`, `Restore`, `UpdateMemWalState`.
- **Never run a `Merge` (`merge`, `add_columns`) while anything else writes the table.** A `Merge` commit is
  retryable-rejected by any concurrent `Append`, `Update`, `Delete`, `Merge`, `Rewrite`, `DataReplacement`, and the
  computed column data is thrown away (files left for cleanup). A concurrent `Project` (drop column) is incompatible.
- **Never compact while a `Merge` or a column update is in flight**; `Rewrite` loses to `Merge` always and to
  `Update`/`Delete`/`DataReplacement` on the same fragments, and they lose to it.
- "Add a column with merge, update it in place later" — hazards:
  1. Rows appended after (or concurrently with, and committed after) the `Merge` have **no data file for the new
     column**: they read `NULL`. The column must be nullable; a later in-place update has to cover those fragments too.
  2. `fields_modified` takes **field ids**, which are sparse and, in legacy mode, **reusable after a drop**. Look the
     id up from the current schema by name each time; never cache it, never use the column position. A cast makes a
     new field id.
  3. An in-place column update (`REWRITE_COLUMNS`) is positional against a fragment: it is invalid after a compaction
     of that fragment (retryable conflict if concurrent; plainly wrong fragment ids if the plan was made before an
     already-committed compaction). Plan against the version you commit against.
  4. Two column updates on the same fragment from two processes: the page only says `Update`/`Update` is
     "rebaseable" by merging deletion masks; it does not say what happens to two column rewrites of the same field.
     Treat as single-writer.
  5. A column rewrite tombstones (`-2`) the old column in its data file; the old bytes stay until compaction. Repeated
     updates grow the table and the number of data files per fragment.
  6. Changing data on a field withdraws index coverage on that field for the touched fragments (those rows are then
     scanned); with stable row ids it bumps last-updated-at for every row in the fragment.
  7. An `Update` without `REWRITE_COLUMNS` (`REWRITE_ROWS`, the row-moving mode) moves the rows to new fragments:
     row addresses change, and `_rowid` changes too unless the table was created with stable row ids.
- A tag pins a version against cleanup; tag before any cleanup and before relying on an old version.

## Unclear or possibly newer than 13.0.0

- **`Append` after a concurrent `Merge`**: the page lists only `Overwrite`, `Restore`, `UpdateMemWalState` as
  conflicting with `Append`, and 13.0.0 behaves that way (measured). The asymmetry is easy to misread: the `Merge`
  side still fails if it commits second.
- **`UpdateMemWalState` is missing from the `Update` and `Rewrite` conflict lists** while present in `Append`,
  `Delete`, `CreateIndex`, `Merge`, `Project`. The page does not say whether that is intended. `UpdateMemWalState`
  has no compatibility section of its own.
- **`UpdateConfig` conflicts** are listed without saying whether they are retryable or incompatible.
- **`Update` vs `Update` in `REWRITE_COLUMNS` mode on the same fragment/field**: not specified (see hazard 4).
- **`Delete` after `Merge`** is "only if there are overlapping fragments" on the `Delete` side but a `Merge` names
  every fragment, so in practice always.
- **`CompositeOperation` / "Transaction V2" / actions** (`AssertCellsUnchanged`, `AppendFragReuseTransition`,
  `TrimFragReuse`, `data_unchanged`): the note dates reader tolerance to v14.0.0; not in 13.0.0's `LanceOperation`.
- **`DataReplacement.data_change`** and the conditional rules that depend on it: field described as absent "in every
  transaction written before the field existed"; whether 13.0.0 has it is unknown. `DataReplacementGroup` exists.
- **Data overlay files**: the page says "not yet supported in any library", yet pylance 13.0.0 exposes
  `LanceOperation.DataOverlay`, `DataOverlayFile`, `DataOverlayGroup`. Treat as unstable; release builds reject the
  flag unless opted in.
- **Fragment metadata tree** (`FLAG_FRAGMENT_TREE`, `_bt/`): "not yet available in released Lance versions".
- **`FLAG_INLINE_DELETIONS` (bit 19)**, **`FLAG_ROW_ADDR_INDICES` (bit 18)**, **`FLAG_OBJECT_METADATA` (bit 17)**,
  **`FLAG_ZONE_WRITE_SEEDS` (bit 16)**, **`FLAG_NON_REUSABLE_FIELD_IDS` (bit 15)** with `max_allocated_field_id`,
  **`FLAG_INDEPENDENT_COVERING_FIELDS` (bit 13)**: high bits, likely after 13.0.0; 16 and 17 are declared
  unsupported. A table written with any of them by a newer library must be refused by a library that lacks the bit.
- **Clustering definitions** (`Manifest.clustering`, `max_clustering_id`), **fragment metadata map**: depend on the
  unsupported bit 17.
- **Spilled row lineage** (`lance.row_lineage.spill`, bit 11) and **stable row id migration**
  (`Dataset::migrate_to_stable_row_ids`): Rust API named; no Python method seen in 13.0.0.
- **Managed blobs** (`lance.blob.managed`, kinds 4 and 5, `_blobs/`, bit 14): unknown whether 13.0.0 writes them;
  activation is permanent.
- **Table config keys**: `lance.compaction.max_overlays_per_fragment`, `max_source_bytes`, `max_source_rows`,
  `max_source_fragments`, `compaction_mode`, `binary_copy_read_batch_bytes`, `index_remap_mode` may be newer than
  13.0.0; an unrecognized `lance.compaction.*` key is ignored with a warning, so a typo or a too-new key fails
  silently.
- **`_versions/latest_version_hint.json`**: optional; unknown whether 13.0.0 writes it.
- **Cleanup semantics** (what exactly `cleanup_old_versions` removes, how tags block it) are not specified in these
  format pages; they point to the user guide `guide/read_and_write.md#automatic-cleanup`, which was not part of this
  read.
- The high-level Python calls (`update`, `delete`, `merge_insert`) retry internally in 13.0.0 (measured for `update`
  and `delete`); the format pages describe only the commit-layer outcome, so "retryable" does not always surface as
  an error.
