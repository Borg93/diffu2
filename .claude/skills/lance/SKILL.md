---
name: lance
description: Rules for this project's Lance tables (scans.lance, pages.lance, vectors). Use before reading, joining columns into, appending to, indexing or designing any Lance table.
---

# Lance in this project

Installed: pylance 13.0.0, data storage version 2.2. The references were written from the Lance docs at
15.0.0-beta.23 (all of `docs/src/guide` and `docs/src/format`, read 17 Nov 2026); each marks what is newer than 13.
Check a rule against the installed library with a small test before relying on it for something new.

## The design

- One row a scan in `runs/htr/curate/scans.lance`, keyed by the unique string `image`; every check is a column, a
  training set is a filter. The columns asked for and their state: `prototypes/column_state.py`.
- Join through `diffu_curate.table.add_features(table, features, key="image")` and nothing else: new columns by
  `merge`, existing ones in place by `LanceFragment.update_columns` + one `LanceOperation.Update`, then
  `optimize_indices()`. It refuses two feature rows for one key (which one wins is undefined in Lance).
- **Vectors:** a fixed-size-list column costs its full width for a null row (measured: 1 % filled is as large as
  100 %). A vector held for every scan may be a column of the scan table (filtered nearest-neighbour search only
  sees columns of the same table); a vector held for few scans goes in a table beside, keyed by `image`.
  `pa.list_(pa.float16(), n)`; read wide columns with `batch_size` about 1,024.
- Thumbnails: plain `binary` inline. Blob columns only from about 1 MB a value.
- A model's whole answer: `pa.json_()`; `json_get_*` filter but cannot be selected, so typed columns beside it.

## Writing: what may run at once

- `merge` (a new column) fails or is lost against any other concurrent write. Nothing else writes the table then.
- Append and append are compatible: two processes may append to one table. Rows appended after a merge hold null.
- In-place column update and compaction on the same fragments clash: never compact during a join.
- An update drops the changed fragments from that column's index until `optimize_indices()`; appended rows are
  unindexed until then too. Results stay right, only slower.
- `merge_insert` and `update` move changed rows to the end of the table; do not use them for feature columns.

## Indexes

BITMAP for few-valued strings and booleans (`pool`, `mark`, `head_cut`, `vl_*`), BTREE for `image` and scores,
INVERTED for captions (set the language, test spellings with `lance.tokenize`). Compact with
`optimize.compact_files(defer_index_remap=True)` before building or optimizing an index. First vector index to try:
IVF_PQ, then IVF_RQ; without an index search is exact.

## Housekeeping

- Every write is a version; old files stay until `cleanup_old_versions` (tagged versions are kept). Tag before a
  risky change: `scans.lance` has `before-features`.
- Tens of thousands of fragments are fine; long jobs appending 2,000 rows a time compact at their end.
- Keep tables on the local disk. The docs say nothing about NFS; do not write a Lance table on `/mnt/scratch` from
  two processes.
- Not in 13.0.0: `local_fsync`, `CleanupPolicy`, the `strategy` option of `update_columns`, format 2.3.

## References (read the one that fits before a new kind of work)

- [write-evolve-types.md](references/write-evolve-types.md): reading, writing, adding and updating columns, JSON, types
- [vectors-blobs-indexes.md](references/vectors-blobs-indexes.md): vector search, blobs, scalar indexes, distributed writes
- [performance-search-storage.md](references/performance-search-storage.md): batch sizes, compaction, text search, stores
- [table-format.md](references/table-format.md): fragments, transactions and the full conflict rules, versions, tags
- [file-and-index-format.md](references/file-and-index-format.md): encodings, null cost, every index type
