---
name: lance
description: Lance tables of this project (scans.lance, pages.lance, vector tables). Load before joining a column, appending rows, building an index, storing vectors, or designing a table.
---

# Lance in this project

Rules cached from the Lance docs (all of `docs/src/guide` and `docs/src/format` at 15.0.0-beta.23) and from
measurements on the installed pylance. The references mark what is newer than the installed version: confirm a rule
new to you with a ten-line test on a throwaway table.

## Join a column

Every check is a column of the scan table, keyed by the unique string `image`; a training set is a filter.

1. Run the join alone: a new column (`merge`) is lost against any concurrent write to that table.
2. Join through `diffu_curate.table.add_features(table, features, key)`: it adds new columns by `merge`, updates held
   ones in place (`LanceFragment.update_columns`, one `LanceOperation.Update`), then refreshes the indexes.
3. Give `features` one row a key: Lance leaves undefined which of two rows wins, so the helper refuses them.
4. Done when `column_state.py` prints the column's new count and a filter on it returns the rows expected.

Rows appended after a join hold null in the joined column. `merge_insert` and `update` move changed rows to the end
of the table: keep them for row tables, use the helper for feature columns.

## Append rows

Two processes may append to one table at once. A long job appends in batches and compacts once at its end with
`optimize.compact_files(defer_index_remap=True)`; compaction and a join on the same fragments clash, so compact
between joins.

## Vectors

A fixed-size-list column costs its full width for every row, null or not (measured: 1 % filled is as large as
100 %).

- Held for every scan: a column of the scan table, `pa.list_(pa.float16(), n)`. Filtered nearest-neighbour search
  sees only columns of its own table (`scanner(nearest=..., filter=..., prefilter=True)`).
- Held for few scans: a table beside, keyed by `image`.
- Read wide columns with `batch_size` near 1,024. Search is exact without an index; first index IVF_PQ, then IVF_RQ.

## Other column types

- A model's whole answer: `pa.json_()`. `json_get_*` filters only, so the fields filtered on are typed columns too.
- Thumbnails: plain `binary`. Blob columns from about 1 MB a value.

## Indexes

BITMAP for few-valued strings and booleans, BTREE for `image` and scores, INVERTED for captions (try spellings with
`lance.tokenize` first). Updated fragments and appended rows stay out of a column's index until
`optimize.optimize_indices()`: results stay right, filters run slower.

## Versions and place

Every write is a version, and old files stay until `cleanup_old_versions`; tagged versions are kept, so tag before
a risky change. Tables live on the local disk: the docs are silent on NFS.

## References

Read the one that fits before a kind of work new to this file.

- [write-evolve-types.md](references/write-evolve-types.md): read, write, add and update columns, JSON, data types
- [vectors-blobs-indexes.md](references/vectors-blobs-indexes.md): vector search, blobs, scalar indexes, distributed writes
- [performance-search-storage.md](references/performance-search-storage.md): batch sizes, compaction, text search, stores
- [table-format.md](references/table-format.md): fragments, transactions, the full table of which writes clash, tags
- [file-and-index-format.md](references/file-and-index-format.md): encodings, the cost of nulls, every index type
